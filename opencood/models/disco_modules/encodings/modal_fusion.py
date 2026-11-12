import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from opencood.models.disco_modules.layer_gnn import GraphConstructor, Mixprop
from opencood.models.disco_modules.autoencoder_conv import Autoencoder
from opencood.models.disco_modules.encodings.transformer import Transformer


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()
    Args:
        pairwise_t_matrix: torch.tensor
            [B, L, L, 4, 4], B batchsize, L max_cav
        H: num.
            Feature map height
        W: num.
            Feature map width
        discrete_ratio * downsample_rate: num.
            One pixel on the feature map corresponds to the actual physical distance

    Returns:
        affine_matrix: torch.tensor
            [B, L, L, 2, 3]
    """

    affine_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    affine_matrix[...,0,1] = affine_matrix[...,0,1] * H / W
    affine_matrix[...,1,0] = affine_matrix[...,1,0] * W / H
    affine_matrix[...,0,2] = affine_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    affine_matrix[...,1,2] = affine_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    return affine_matrix


class MultiModalFusion(nn.Module):
    def __init__(self, dim, mode='implicit', ratio=0.8, num_layers=3, threshold=0.1):
        super().__init__()
        self.dim =  256
        self.ratio = ratio
        self.mode = mode
        if self.mode == 'implicit':
            self.threshold = 0.5
            self.value_func = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Sigmoid(),
            )

        else:
            self.threshold = 0.1
            self.autoencoder = Autoencoder()
            self.rec_loss = nn.MSELoss()
            self.abs_loss = nn.L1Loss()

            self.s_func = nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.AdaptiveAvgPool2d((128, 128)),
            )

            self.v_func = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.d_func = nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.AdaptiveAvgPool2d((256, 256)),
            )

        self.gc = GraphConstructor(dim=dim, alpha=3)
        self.num_layers = num_layers
        self.gconv1, self.gconv2, self.norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.gconv1.append(Mixprop(dim, dim, gdep=2, dropout=0.3, alpha=0.05))
            self.gconv2.append(Mixprop(dim, dim, gdep=2, dropout=0.3, alpha=0.05))
            self.norm.append(nn.InstanceNorm2d(dim))

        self.skipE = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, feats, training):
        # 模态融合 img, pc, radar: B*C*Y*X

        con_feat = torch.stack(feats, dim=1)
        B, M, C, H, W = con_feat.shape
        con_feat = rearrange(con_feat, 'b m c h w -> (b m) c h w')

        auto_enc_loss, svd_loss = torch.tensor(0.0, requires_grad=True).to(con_feat.device), torch.tensor(0.0, requires_grad=True).to(con_feat.device)
        if self.mode == 'implicit':
            feat_v = self.value_func(con_feat).squeeze(-1).squeeze(-1)
        
        else:
            feat_mid, feat_rec = self.autoencoder(con_feat)
            # b*c*c, b*c, b*n*n
            feat_v = self.v_func(feat_mid).flatten(1)
            
            if training:
                feat_s, feat_d = self.s_func(feat_mid).squeeze(1), self.d_func(feat_mid).squeeze(1)
                # construct diag matrix
                diag_v = torch.zeros((B*M, feat_s.shape[1], feat_d.shape[1])).to(con_feat.device)
                diag_v[:, :min(feat_s.shape[1], feat_d.shape[1]), :min(feat_s.shape[1], feat_d.shape[1])] = torch.diag_embed(feat_v)
                # recover matrix
                rec_feat_mid = torch.bmm(feat_s, torch.bmm(diag_v, feat_d))
                auto_enc_loss = self.rec_loss(con_feat, feat_rec)
                svd_loss = self.abs_loss(feat_mid.flatten(2), rec_feat_mid)
        
        # count principal components in channel dimension
        feat_v = rearrange(feat_v, '(b m) c -> b m c', b=B, m=M)
        counts = torch.sum(feat_v > self.threshold, dim=-1)
        best_indices = torch.argmax(counts, dim=-1)
        # best_indices = torch.randint(low=0, high=2, size=(B,)).to(con_feat.device)

        con_feat = rearrange(con_feat, '(b m) c h w -> b m c h w', b=B, m=M)
        x = con_feat
        for i in range(self.num_layers):
            Adj = self.gc(x)
            residuals = self.gconv1[i](x, Adj) + self.gconv2[i](x, Adj.transpose(1,2))
            x = x + residuals
            x = x.view(B*M, C, H, W)
            x = self.norm[i](x)
            x = x.view(B, M, C, H, W)

        x = x.view(B*M, C, H, W)
        con_feat = con_feat.view(B*M, C, H, W)
        
        con_feat = con_feat + self.skipE(x)
        con_feat = F.relu(con_feat)
        con_feat = self.conv(con_feat)
        con_feat = con_feat.view(B, M, C, H, W)
        
        fused_feat = []
        for idx, index in enumerate(best_indices):
            fused_feat.append(con_feat[idx,index,:,:,:])
        fused_feat = torch.stack(fused_feat, dim=0)
        return fused_feat, auto_enc_loss, svd_loss


class ModalFusionBlock(nn.Module):   
    def __init__(self, dim, mode='explicit', modality_num=1):
        super(ModalFusionBlock, self).__init__()   
        self.mode = mode
        if mode in ['explicit', 'implicit']:
            self.fusion = MultiModalFusion(dim, mode)
        elif mode in ['mce']:
            self.fusion = MCEBlock(dim, modality_num)
        else:
            self.fusion = nn.Conv2d(dim*modality_num, dim, kernel_size=1)

    def forward(self, modal_features, mode=[1,1], training=True):
        # justify
        selected_modal_features = []
        
        if self.mode in ['explicit', 'implicit']:
            for i in range(len(mode)):
                if mode[i] > 0:
                    selected_modal_features.append(modal_features[i])
            
            x, rec_loss, svd_loss = self.fusion(selected_modal_features, training=training)
            # x, rec_loss, svd_loss = self.mask_modality(selected_modal_features[0], selected_modal_features[1], batch_dict['record_len'])
            return x, rec_loss, svd_loss

        elif self.mode == 'mce':
            for i in range(len(mode)):
                if mode[i] > 0:
                    selected_modal_features.append(modal_features[i])
                else:
                    # drop the modality
                    miss_features = torch.zeros_like(modal_features[i])
                    selected_modal_features.append(miss_features)
            x  = self.fusion(selected_modal_features)

            enc_loss = torch.tensor(0.0, requires_grad=True).to(selected_modal_features[0].device)
            svd_loss = torch.tensor(0.0, requires_grad=True).to(selected_modal_features[0].device)
            return x, enc_loss, svd_loss

        # CoVQM
        for i in range(len(mode)):
            if mode[i] > 0:
                selected_modal_features.append(modal_features[i])
            else:
                # drop the modality
                miss_features = torch.zeros_like(modal_features[i])
                selected_modal_features.append(miss_features)
        enc_loss = torch.tensor(0.0, requires_grad=True).to(selected_modal_features[0].device)
        svd_loss = torch.tensor(0.0, requires_grad=True).to(selected_modal_features[0].device)
        # process lidar to get bev
        return self.fusion(torch.cat(selected_modal_features, dim=1)), enc_loss, svd_loss

    def mask_modality(self, x, y, record_len):
        """
        x: lidar feature shape (M, C, H, W)
        y: image feature shape (M, C, H, W)
        """
        
        ego_lidar = x[0:1,]
        ego_image = y[0:1,]

        # # 1. L + C
        # if y.shape[0] > 1:
        #     rec_feature = torch.cat([ego_lidar, y[1:2]], dim=0)
        # else:
        #     rec_feature = ego_lidar
        # return self.fusion([rec_feature], training=False)
        
        # # 2. C + L
        # if x.shape[0] > 1:
        #     rec_feature = torch.cat([ego_image, x[1:2]], dim=0)
        # else:
        #     rec_feature = ego_image
        # return self.fusion([rec_feature], training=False)
        
        # # 3. LC + C
        # ego_feature, rec_loss, svd_loss = self.fusion([ego_lidar, ego_image], training=False)
        # agent_features = [ego_feature]
        # if y.shape[0] > 1:
        #     nearby_feature, rec_loss2, svd_loss2 = self.fusion([y[1:2]], training=False)
        #     agent_features.append(nearby_feature)
        #     rec_loss = rec_loss + rec_loss2
        #     svd_loss = svd_loss + svd_loss2
        # return torch.cat(agent_features, dim=0), rec_loss, svd_loss

        # # 4. LC + L
        # ego_feature, rec_loss, svd_loss = self.fusion([ego_lidar, ego_image], training=False)
        # agent_features = [ego_feature]
        # if x.shape[0] > 1:
        #     nearby_feature, rec_loss2, svd_loss2 = self.fusion([x[1:2]], training=False)
        #     agent_features.append(nearby_feature)
        #     rec_loss = rec_loss + rec_loss2
        #     svd_loss = svd_loss + svd_loss2
        # return torch.cat(agent_features, dim=0), rec_loss, svd_loss

        # # 5. L + LC
        # ego_feature, rec_loss, svd_loss = self.fusion([ego_lidar], training=False)
        # agent_features = [ego_feature]
        # if x.shape[0] > 1:
        #     nearby_feature, rec_loss2, svd_loss2 = self.fusion([x[1:2], y[1:2]], training=False)
        #     agent_features.append(nearby_feature)
        #     rec_loss = rec_loss + rec_loss2
        #     svd_loss = svd_loss + svd_loss2
        # return torch.cat(agent_features, dim=0), rec_loss, svd_loss

        # # 6. C + LC
        # ego_feature, rec_loss, svd_loss = self.fusion([ego_image], training=False)
        # agent_features = [ego_feature]
        # if x.shape[0] > 1:
        #     nearby_feature, rec_loss2, svd_loss2 = self.fusion([x[1:2], y[1:2]], training=False)
        #     agent_features.append(nearby_feature)
        #     rec_loss = rec_loss + rec_loss2
        #     svd_loss = svd_loss + svd_loss2
        # return torch.cat(agent_features, dim=0), rec_loss, svd_loss  


class MCEBlock(nn.Module):   
    def __init__(self, dim, modality_num=1):
        super(MCEBlock, self).__init__()
        self.positional_encodings = []
        for i in range(modality_num):
            self.positional_encodings.append(nn.Parameter(torch.zeros(1, 1, dim)))
        self.inter = Transformer(dim, mlp_dim=dim*4)
        self.fusion = nn.Conv2d(dim*modality_num, dim, kernel_size=1)

    def forward(self, modal_features):
        B, C, H, W = modal_features[0].shape
        stacked_features = torch.stack(modal_features, dim=1).view(B, len(modal_features), C, -1)
        # [B, L, M, C]
        stacked_features = rearrange(stacked_features, 'b m c l -> b l m c')
        # [1, 1, M, C]
        stacked_pos = torch.stack(self.positional_encodings, dim=-2).to(stacked_features.device)
        # justify
        stacked_features_rec = self.inter(stacked_features, stacked_pos)
        # [B, L, M, C] -> [B, M, C, H, W]
        stacked_features_rec = rearrange(stacked_features_rec, 'b l m c -> b m c l')
        stacked_features_rec = stacked_features_rec.view(B, len(modal_features), C, H, W)
        concat_features = []
        for i in range(len(modal_features)):
            concat_features.append(stacked_features_rec[:, i])
        return self.fusion(torch.cat(concat_features, dim=1))
