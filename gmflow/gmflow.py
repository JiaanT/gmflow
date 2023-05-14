import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position, feature_remove_position_single
from .deformable import OpticalFlowDeformableTransformer, build_optical_flow_deformable_transformer


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # # Transformer
        # self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
        #                                       d_model=feature_channels,
        #                                       nhead=num_head,
        #                                       attention_type=attention_type,
        #                                       ffn_dim_expansion=ffn_dim_expansion,
        #                                       )
        
        self.deformable_transformer = OpticalFlowDeformableTransformer(d_model=128, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=num_scales, dec_n_points=4,  enc_n_points=4,
                 two_stage=False)
        
        # # Use [b, n, c] to predict flow
        # self.flow_predictor = nn.Linear(feature_channels, 2)
        
        self.flow_head_1 = nn.Linear(2 * feature_channels, 64)  # You can adjust the hidden size
        self.flow_head_2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

        # # flow propagation with self-attn
        # self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # # convex upsampling: concat feature0 and flow as input
        # self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow
    
    
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    

    def forward(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        
        
        # GMFlow Transformer
        
        pos_embeds = []
        # feature1_pure_list = []

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]


            attn_splits = attn_splits_list[scale_idx]

            # add position to features
            feature0, feature1, position = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            pos_embeds.append(position)

            # Transformer
            # feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            # feature1_pure = feature_remove_position_single(feature1, position, attn_splits)
            # feature1_pure_list.append(feature1_pure)
        
        
        # Construct src
        
        # sub_feature0_list = []
        # # reference_points = []
        
        # # Initialize reference_points_list
        # reference_points_list = []

        # for scale_idx in range(self.num_scales):
        #     feature0 = feature0_list[scale_idx]  # [B, C, H, W]
        #     # input_feat_map is a tensor of shape [B, C, H, W]
        #     B, C, H, W = feature0.shape

        #     # Compute the new height and width for the sampled sub_feature_map
        #     new_H, new_W = int(H * 0.8), int(W * 0.8)

        #     # Create the sampling grid
        #     grid_W = torch.linspace(-0.8, 0.8, new_W)
        #     grid_H = torch.linspace(-0.8, 0.8, new_H)
        #     sampling_grid = torch.stack(torch.meshgrid(grid_H, grid_W), dim=-1).to(feature0.device)
        #     sampling_grid = sampling_grid.view((1, new_H, new_W, 2)).float()

        #     # Apply grid sampling
        #     sub_feature0 = F.grid_sample(feature0, sampling_grid, mode='nearest', align_corners=True)
        #     sub_feature0_list.append(sub_feature0)

        #     # Normalize the sampling_grid to be in the range (0, 1)
        #     normalized_sampling_grid = (sampling_grid + 1) / 2

        #     # Scale the normalized sampling grid by the original feature map's height and width
        #     reference_points = normalized_sampling_grid * torch.tensor([H - 1, W - 1], dtype=torch.float32, device=feature0.device)

        #     # Reshape the reference_points tensor to the shape (N, L, 2)
        #     N = new_H * new_W
        #     reference_points = reference_points.view(1, N, 2)

        #     # Append reference_points to the reference_points_list
        #     reference_points_list.append(reference_points)

        # # Stack the reference_points_list along the second dimension to create a tensor of shape (N, P, L, 2)
        # # reference_points_tensor = torch.cat(reference_points_list, dim=1)
        
        
        
        B, _, img_H, img_W = img0.shape

        # Calculate the grid size based on 80% of the original image size
        new_H, new_W = int(img_H * 0.0625), int(img_W * 0.0625)

        # Create the sampling grid
        grid_W = torch.linspace(-1, 1, new_W)
        grid_H = torch.linspace(-1, 1, new_H)
        sampling_grid = torch.stack(torch.meshgrid(grid_H, grid_W), dim=-1).to(img0.device)
        sampling_grid = sampling_grid.view((1, new_H, new_W, 2)).float()

        sub_feature0_list = []
        reference_points_list = []

        for scale_idx in range(self.num_scales):
            feature0 = feature0_list[scale_idx]  # [B, C, H, W]
            B, C, H, W = feature0.shape

            # Apply grid sampling with the same grid size for all feature maps
            sub_feature0 = F.grid_sample(feature0, sampling_grid, mode='nearest', align_corners=True)
            sub_feature0_list.append(sub_feature0)
            # TODO: Construct reference points, just like before, ask GPT
            # Normalize the sampling_grid to be in the range (0, 1)
            normalized_sampling_grid = (sampling_grid + 1) / 2

            # Reshape the reference_points tensor to the shape (N, L, 2)
            N = new_H * new_W
            reference_points = normalized_sampling_grid.view(1, N, 2)

            # Append reference_points to the reference_points_list
            reference_points_list.append(reference_points)
            
            
        # Concat reference_points_list
        reference_points_flat = torch.cat(reference_points_list, dim=1)  # Shape: (B, N, 2)
        
        # concated_reference_points = concated_reference_points[:, :, None] * valid_ratios[:, None]

        # # Stack the tensors in the reference_points_list
        # stacked_reference_points = torch.stack(reference_points_list, dim=3)  # Shape: (B, N, 2, L)

        # # Rearrange the dimensions to the desired order (B, N, L, 2)
        # stacked_reference_points = stacked_reference_points.permute(0, 1, 3, 2)  # Shape: (B, N, L, 2)
            
            
        # # Construct reference points
        
        # spatial_shapes = []
        # for feature in feature1_list:
        #     _, _, height, width = feature.shape
        #     spatial_shapes.append((height, width))
            
        # # Extract the spatial shapes (H and W) from each level of the pyramid feature maps
        # spatial_shapes = [feature_map.shape[2:] for feature_map in feature1_list]

        # # Construct the valid_ratios tensor with the ratio 0.8 for both height and width
        # L = len(spatial_shapes)  # Number of target levels
        # B = feature1_list[0].shape[0]  # Batch size
        # valid_ratios = torch.full((B, L, 2), 0.8).to(feature1_list[0].device)
        
        # reference_points = self.get_reference_points(spatial_shapes, valid_ratios=valid_ratios, device=feature1.device)

        
        srcs = sub_feature0_list
        tgts = feature1_list
        
        
        # Prepare input for encoder
        src_masks = [torch.zeros((src.shape[0], src.shape[2], src.shape[3]), dtype=torch.bool, device=src.device) for src in sub_feature0_list] #TODO: check
        
        src_flatten = []
        src_mask_flatten = []
        src_spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, src_masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            src_spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            src_mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        src_mask_flatten = torch.cat(src_mask_flatten, 1)
        src_spatial_shapes = torch.as_tensor(src_spatial_shapes, dtype=torch.long, device=src_flatten.device)
        src_level_start_index = torch.cat((src_spatial_shapes.new_zeros((1,)), src_spatial_shapes.prod(1).cumsum(0)[:-1]))
        src_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in src_masks], 1)
        
        
        reference_points_flat = reference_points_flat[:, :, None] * src_valid_ratios[:, None]
        
        
        # Prepare input for decoder: construct spatial_shapes and mask_flatten of tgt_flatten
        tgt_masks = [torch.zeros((tgt.shape[0], tgt.shape[2], tgt.shape[3]), dtype=torch.bool, device=tgt.device) for tgt in tgts] #TODO: check
        
        tgt_flatten = []
        tgt_spatial_shapes = []
        tgt_mask_flatten = []
        for lvl, (tgt, mask) in enumerate(zip(tgts, tgt_masks)):
            bs, c, h, w = tgt.shape
            tgt = tgt.flatten(2).transpose(1, 2)
            tgt_flatten.append(tgt)
            spatial_shape = (h, w)
            tgt_spatial_shapes.append(spatial_shape)
            mask = mask.flatten(1)
            tgt_mask_flatten.append(mask)
        tgt_flatten = torch.cat(tgt_flatten, 1)
        tgt_mask_flatten = torch.cat(tgt_mask_flatten, 1)
        tgt_spatial_shapes = torch.as_tensor(tgt_spatial_shapes, dtype=torch.long, device=src_flatten.device)
        tgt_level_start_index = torch.cat((tgt_spatial_shapes.new_zeros((1,)), tgt_spatial_shapes.prod(1).cumsum(0)[:-1]))
        tgt_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in tgt_masks], 1)
        
        
    
        extracted_tgt = self.deformable_transformer(
            src_flatten, src_spatial_shapes, src_level_start_index, src_mask_flatten, src_valid_ratios,
            tgt_flatten, tgt_spatial_shapes, tgt_level_start_index, tgt_mask_flatten, tgt_valid_ratios,
            reference_points_flat
        )
        
        concat_tensor = torch.cat((src_flatten, extracted_tgt), dim=-1)


        # Reshape the tensor to be [B*N, 2C] because nn.Linear expects the last dimension to be the feature dimension
        reshaped_tensor = concat_tensor.view(B*src_flatten.shape[1], 2*src_flatten.shape[2])

        # Pass through the flow head
        flow = self.flow_head_2(self.relu(self.flow_head_1(reshaped_tensor)))

        # Reshape the output to be [B, N, 2]
        flow = flow.view(B, src_flatten.shape[1], 2)
        
        
        
            
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            
            
            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
            
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]
            
            
            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict
