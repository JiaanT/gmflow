from __future__ import division
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch_scatter
from torch_sparse import coalesce


pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    global pixel_coords_init
    global pixel_coords_ref

    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    pixel_coords_init = torch.stack((j_range, i_range), dim=1)  # [1, 2, H, W]
    pixel_coords_ref = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(pixel_coords, depth, intrinsics_inv):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords.double()).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)




def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    # mask coords with depth=0
    X_norm[Z == proj_c2p_tr[:, 2]] = 2
    Y_norm[Z == proj_c2p_tr[:, 2]] = 2
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def forward_warp(src_img, src_depth, pose_src2ref, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the reference image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(src_img, 'img', 'B3HW') # src_frame
    check_sizes(src_depth, 'depth', 'B1HW')
    check_sizes(pose_src2ref, 'pose', 'B44')
    
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = src_img.size()

    set_id_grid(src_depth.squeeze(1))
    global pixel_coords
    global pixel_coords_init
    global pixel_coords_ref

    # Get projection matrix for tgt camera frame to source pixel frame
    cam_coords = pixel2cam(pixel_coords, src_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    proj_pixel_src2ref = intrinsics @ pose_src2ref[:, :3].double()  # [B, 3, 4]
    rot, tr = proj_pixel_src2ref[:, :, :3], proj_pixel_src2ref[:, :, -1:]
    pixel_coords_src2ref, computed_depth_src2ref = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]

    size = [computed_depth_src2ref.shape[2], computed_depth_src2ref.shape[3]]
    # size[0] *= 1000; size[1] *= 1000

    coords_sparse = pixel_coords_src2ref.reshape(-1,2).permute(1, 0)[[1,0]]
    value_sparse = computed_depth_src2ref.reshape(-1)
    valid_mask = (coords_sparse[0] != 2) & (coords_sparse[1] != 2)
    coords_sparse = coords_sparse[:, valid_mask]
    value_sparse = value_sparse[valid_mask]

    coords_sparse[0] = (coords_sparse[0] + 1) / 2 * (img_height-1)
    coords_sparse[1] = (coords_sparse[1] + 1) / 2 * (img_width-1)

    # coords_sparse *= 1000
    
    depth_sparse = torch.sparse_coo_tensor(coords_sparse.long(), value_sparse, size=size)

    # depth_sparse.sparse_resize_((img_height, img_width), sparse_dim=depth_sparse.sparse_dim(), dense_dim=depth_sparse.dense_dim())
    computed_depth = depth_sparse.to_dense()[None, None, ...]


    # depth_w, fw_val = [], []
    # for coo, z in zip(pixel_coords_src2ref, computed_depth_src2ref):
    #     idx = coo.reshape(-1,2).permute(1,0)[[1,0]]
    #     val = z.reshape(-1)

    #     # filter invalid coords
    #     valid_mask = (idx[0] != 2) & (idx[1] != 2)
    #     idx = idx[:, valid_mask]
    #     val = val[valid_mask]

    #     hh = src_depth.shape[2]; ww = src_depth.shape[3]
    #     idx[0] = (idx[0] + 1) / 2 * (hh-1)
    #     idx[1] = (idx[1] + 1) / 2 * (ww-1)

    #     _idx, _val = coalesce(idx.long(), 1/val, m=hh, n=ww, op='mean')
    #     print(_idx.shape)
    #     depth_w.append( 1/torch.sparse.FloatTensor(_idx, _val, torch.Size([hh,ww])).to_dense() )
    #     fw_val.append( 1- (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh,ww])).to_dense()==0).float() )
    #     # pdb.set_trace()
    # depth_w = torch.stack(depth_w, dim=0)
    # fw_val = torch.stack(fw_val, dim=0)
    # # depth_w[fw_val==0] = 0
    # print(depth_w.max())
    # computed_depth = F.interpolate(depth_w[None, ...], (computed_depth_src2ref.shape[2], computed_depth_src2ref.shape[3]), mode='bilinear')

    cam_coords_ref = pixel2cam(pixel_coords_ref, computed_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    proj_pixel_ref2src = intrinsics @ pose_src2ref.double().inverse()[:, :3]  # [B, 3, 4]
    rot, tr = proj_pixel_ref2src[:, :, :3], proj_pixel_ref2src[:, :, -1:]
    pixel_coords_ref2src, _ = cam2pixel2(cam_coords_ref, rot, tr, padding_mode)  # [B,H,W,2]

    pixel_coords_ref2src[pixel_coords_ref2src.abs() > 1] = 2 # invalid coords
    img_src2ref = F.grid_sample(src_img.double(), pixel_coords_ref2src, padding_mode=padding_mode, align_corners=False) # ref frame

    valid_points = pixel_coords_ref2src.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    pixel_coords_ref2src = pixel_coords_ref2src.permute(0, 3, 1, 2)
    flow = pixel_coords_ref2src - pixel_coords_init # flow_ref2src

    return img_src2ref, computed_depth, flow, valid_mask

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    # src to ref

    img_src_numpy = cv2.imread('/mnt/nas/share/home/xugk/data/scannet_test/scene0707_00/color/5.jpg')
    img_src_numpy = cv2.resize(img_src_numpy, (640, 480), cv2.INTER_LINEAR)
    depth_src_numpy = cv2.imread('/mnt/nas/share/home/xugk/data/scannet_test/scene0707_00/depth/5.png', -1) / 1000.
    pose_src = np.loadtxt('/mnt/nas/share/home/xugk/data/scannet_test/scene0707_00/pose/5.txt').reshape(4, 4)
    pose_ref = np.loadtxt('/mnt/nas/share/home/xugk/data/scannet_test/scene0707_00/pose/0.txt').reshape(4, 4)
    intrinsic = np.loadtxt('/mnt/nas/share/home/xugk/data/scannet_test/scene0707_00/intrinsic/intrinsic_depth.txt')[:3, :3]
    
    img_src = torch.from_numpy(img_src_numpy).permute(2, 0, 1)[None, ...] / 255.
    depth_src = torch.from_numpy(depth_src_numpy)[None, None, ...]
    pose_src2ref = torch.from_numpy(np.linalg.inv(pose_ref) @ pose_src)[None, ...]
    intrinsic = torch.from_numpy(intrinsic)[None, ...]

    projected_img, computed_depth, flow, valid_mask = forward_warp(img_src, depth_src, pose_src2ref, intrinsic)

    projected_img_viz = projected_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * 255
    cv2.imwrite('temp_rgb.png', projected_img_viz)
    plt.imsave('temp_depth_computed.png', computed_depth.detach().cpu().numpy()[0, 0], cmap='rainbow')
    plt.imsave('temp_depth_src.png', depth_src_numpy, cmap='rainbow')
    cv2.imwrite('temp_mask.png', valid_mask.detach().cpu().numpy()[0, 0] * 255)

    img_ref = cv2.imread('/mnt/nas/share/home/xugk/data/scannet_test/scene0707_00/color/0.jpg')
    img_ref = cv2.resize(img_ref, (640, 480), cv2.INTER_LINEAR)
    cv2.imwrite('temp_rgb_ref.png', img_ref)
    cv2.imwrite('temp_rgb_src.png', img_src_numpy)
