import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pareconv.modules.layers import VNLinear, VNLinearLeakyReLU, VNLeakyReLU, VNStdFeature
from pareconv.modules.ops import index_select

EPS = 1e-8


class CorrelationNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=(8, 8), last_bn=False, temp=1.0):
        super().__init__()
        self.vn_layer = VNLinearLeakyReLU(
            in_channel,
            out_channel * 2,
            dim=4,
            share_nonlinearity=False,
            negative_slope=0.2,
        )
        self.last_bn = last_bn
        self.temp = temp
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()

        hidden_unit = [] if hidden_unit is None else copy.deepcopy(list(hidden_unit))
        hidden_unit.insert(0, out_channel * 2)
        hidden_unit.append(out_channel)
        for i in range(1, len(hidden_unit)):
            is_last = i == len(hidden_unit) - 1
            self.mlp_convs_hidden.append(
                nn.Conv1d(
                    hidden_unit[i - 1],
                    hidden_unit[i],
                    kernel_size=1,
                    bias=not last_bn if is_last else False,
                )
            )
            if not is_last or last_bn:
                self.mlp_bns_hidden.append(nn.BatchNorm1d(hidden_unit[i]))

    def forward(self, xyz):
        scores = self.vn_layer(xyz)
        scores = torch.norm(scores, p=2, dim=2)
        for i, conv in enumerate(self.mlp_convs_hidden):
            is_last = i == len(self.mlp_convs_hidden) - 1
            scores = conv(scores)
            if not is_last:
                scores = F.relu(self.mlp_bns_hidden[i](scores), inplace=True)
            elif self.last_bn:
                scores = self.mlp_bns_hidden[i](scores)
        return F.softmax(scores / self.temp, dim=1)


class PARE_Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, share_nonlinearity=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(
            in_channel=3,
            out_channel=kernel_size,
            hidden_unit=[kernel_size],
        )

        conv_dim = in_dim + 2
        weight = nn.init.kaiming_normal_(
            torch.empty(kernel_size, conv_dim, out_dim // 2)
        ).contiguous()
        weight = weight.permute(1, 0, 2).reshape(conv_dim, kernel_size * out_dim // 2)
        self.weightbank = nn.Parameter(weight)

        self.relu = VNLeakyReLU(out_dim // 2, share_nonlinearity)
        self.unary = VNLinearLeakyReLU(out_dim // 2, out_dim)

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        n_points, n_neighbors = neighbor_indices.shape
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)
        centers = pts.mean(-1, keepdim=True).expand(-1, -1, -1, n_neighbors)
        cross = torch.cross(pts, centers, dim=2)
        local_feats = torch.cat([pts, centers, cross], dim=1)

        scores = self.score_net(local_feats)
        projected = torch.einsum('ncdk,cf->nfdk', local_feats, self.weightbank)
        projected = projected.reshape(n_points, self.kernel_size, -1, 3, n_neighbors)
        projected = (projected * scores[:, :, None, None]).sum(dim=1)

        new_feats = F.normalize(projected, p=2, dim=2).mean(dim=-1)
        new_feats = self.relu(new_feats)
        return self.unary(new_feats)


class PARE_Conv_Resblock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        shortcut_linear=False,
        share_nonlinearity=False,
        conv_info=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(
            in_channel=3,
            out_channel=kernel_size,
            hidden_unit=[kernel_size],
        )

        self.conv_way = conv_info['conv_way']
        self.use_xyz = conv_info['use_xyz']
        conv_dim = in_dim * 2 if self.conv_way == 'edge_conv' else in_dim
        if self.use_xyz:
            conv_dim += 1

        weight = nn.init.kaiming_normal_(
            torch.empty(kernel_size, conv_dim, out_dim // 2)
        ).contiguous()
        weight = weight.permute(1, 0, 2).reshape(conv_dim, kernel_size * out_dim // 2)
        self.weightbank = nn.Parameter(weight)

        self.relu = VNLeakyReLU(out_dim // 2, share_nonlinearity)
        self.shortcut_proj = VNLinear(in_dim, out_dim) if shortcut_linear else nn.Identity()
        self.unary = VNLinearLeakyReLU(out_dim // 2, out_dim)

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        n_points, n_neighbors = neighbor_indices.shape
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)
        centers = pts.mean(-1, keepdim=True).expand(-1, -1, -1, n_neighbors)
        cross = torch.cross(pts, centers, dim=2)
        local_feats = torch.cat([pts, centers, cross], dim=1)
        scores = self.score_net(local_feats)

        neighbor_feats = s_feats[neighbor_indices].permute(0, 2, 3, 1)
        identity = self.shortcut_proj(neighbor_feats[..., 0])

        if self.conv_way == 'edge_conv':
            query_feats = neighbor_feats[..., 0:1]
            neighbor_feats = torch.cat([neighbor_feats - query_feats, neighbor_feats], dim=1)
        if self.use_xyz:
            neighbor_feats = torch.cat([neighbor_feats, pts], dim=1)

        projected = torch.einsum('ncdk,cf->nfdk', neighbor_feats, self.weightbank)
        projected = projected.reshape(n_points, self.kernel_size, -1, 3, n_neighbors)
        projected = (projected * scores[:, :, None, None]).sum(dim=1)

        new_feats = F.normalize(projected, p=2, dim=2).mean(dim=-1)
        new_feats = self.relu(new_feats)
        new_feats = self.unary(new_feats)
        return new_feats + identity


class EquivariantLayerScale(nn.Module):
    pass

    def __init__(self, channels, init_value=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, channels, 1), float(init_value)))

    def forward(self, x):
        return self.gamma * x


class RotationEquivariantSPSA(nn.Module):
    pass






    def __init__(
        self,
        channels,
        partial_ratio=0.5,
        qk_channels=32,
        ffn_ratio=2.0,
        distance_hidden=16,
        share_nonlinearity=False,
    ):
        super().__init__()
        if not 0.0 < partial_ratio < 1.0:
            raise ValueError('partial_ratio must be in (0, 1).')

        self.attn_channels = max(1, int(round(channels * partial_ratio)))
        self.identity_channels = channels - self.attn_channels
        if self.identity_channels < 1:
            raise ValueError('partial_ratio leaves no identity channels.')

        self.qk_channels = min(int(qk_channels), self.attn_channels)
        self.query_proj = VNLinear(self.attn_channels, self.qk_channels)
        self.key_proj = VNLinear(self.attn_channels, self.qk_channels)
        self.value_proj = VNLinear(self.attn_channels, self.qk_channels)
        self.context_proj = VNLinear(self.qk_channels, self.attn_channels)

        self.distance_bias = nn.Sequential(
            nn.Linear(1, distance_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(distance_hidden, 1),
        )

        expanded = max(self.attn_channels, int(round(self.attn_channels * ffn_ratio)))
        self.ffn = nn.Sequential(
            VNLinearLeakyReLU(
                self.attn_channels,
                expanded,
                dim=3,
                share_nonlinearity=share_nonlinearity,
            ),
            VNLinear(expanded, self.attn_channels),
        )
        self.output_proj = VNLinear(channels, channels)

    def forward(self, feats, points, neighbor_indices):
        if feats.ndim != 3 or feats.shape[-1] != 3:
            raise ValueError(f'feats must have shape (N, C, 3), got {tuple(feats.shape)}')
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f'points must have shape (N, 3), got {tuple(points.shape)}')
        if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != points.shape[0]:
            raise ValueError('neighbor_indices must have shape (N, K).')

        identity_feats = feats[:, : self.identity_channels]
        attn_feats = feats[:, self.identity_channels :]

        query = self.query_proj(attn_feats)
        key = self.key_proj(attn_feats)
        value = self.value_proj(attn_feats)

        neighbor_key = key[neighbor_indices]
        neighbor_value = value[neighbor_indices]

        logits = (query[:, None] * neighbor_key).sum(dim=(-1, -2))
        logits = logits / math.sqrt(float(self.qk_channels * 3))

        relative_points = points[neighbor_indices] - points[:, None]
        squared_distance = (relative_points * relative_points).sum(dim=-1)
        normalized_distance = squared_distance / (
            squared_distance.mean(dim=1, keepdim=True) + EPS
        )
        logits = logits + self.distance_bias(normalized_distance.unsqueeze(-1)).squeeze(-1)

        attention = torch.softmax(logits, dim=1)
        context = (attention[..., None, None] * neighbor_value).sum(dim=1)
        context = self.context_proj(context)

        enhanced = attn_feats + context
        delta_attn = context + self.ffn(enhanced)
        delta = torch.cat([torch.zeros_like(identity_feats), delta_attn], dim=1)
        return self.output_proj(delta)


class EquivariantNeighborhoodKernel(nn.Module):
    pass

    def __init__(self, channels, branch_channels, radial_hidden=16, share_nonlinearity=False):
        super().__init__()
        self.input_proj = VNLinear(channels, branch_channels)
        local_channels = 2 * branch_channels + 1
        self.local_mlp = nn.Sequential(
            VNLinearLeakyReLU(
                local_channels,
                branch_channels,
                dim=4,
                share_nonlinearity=share_nonlinearity,
            ),
            VNLinearLeakyReLU(
                branch_channels,
                branch_channels,
                dim=4,
                share_nonlinearity=share_nonlinearity,
            ),
        )
        self.radial_score = nn.Sequential(
            nn.Linear(1, radial_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(radial_hidden, 1),
        )

    def forward(self, feats, points, neighbor_indices):
        if neighbor_indices.shape[1] < 1:
            raise ValueError('Each PKI branch requires at least one neighbor.')
        projected_feats = self.input_proj(feats)
        neighbor_feats = projected_feats[neighbor_indices]
        center_feats = projected_feats[:, None]
        edge_feats = neighbor_feats - center_feats
        relative_points = points[neighbor_indices] - points[:, None]

        local_feats = torch.cat(
            [edge_feats, neighbor_feats, relative_points[:, :, None, :]],
            dim=2,
        )
        local_feats = local_feats.permute(0, 2, 3, 1).contiguous()
        local_feats = self.local_mlp(local_feats)

        squared_distance = (relative_points * relative_points).sum(dim=-1)
        normalized_distance = squared_distance / (
            squared_distance.mean(dim=1, keepdim=True) + EPS
        )
        radial_logits = self.radial_score(normalized_distance.unsqueeze(-1)).squeeze(-1)
        radial_weights = torch.softmax(radial_logits, dim=1)
        return (local_feats * radial_weights[:, None, None, :]).sum(dim=-1)


class RotationEquivariantMSPKI(nn.Module):
    pass

    def __init__(
        self,
        channels,
        neighbor_scales=(8, 16, 32),
        branch_channels=16,
        ffn_ratio=2.0,
        share_nonlinearity=False,
        fusion_mode='learned',
    ):
        super().__init__()




        scales = tuple(sorted(int(scale) for scale in neighbor_scales))
        if not scales or scales[0] < 1:
            raise ValueError('neighbor_scales must contain positive integers.')
        self.neighbor_scales = scales
        if fusion_mode not in ('learned', 'equal'):
            raise ValueError("fusion_mode must be 'learned' or 'equal'.")
        self.fusion_mode = fusion_mode

        self.identity_branch = VNLinear(channels, branch_channels)
        self.scale_branches = nn.ModuleList(
            [
                EquivariantNeighborhoodKernel(
                    channels,
                    branch_channels,
                    share_nonlinearity=share_nonlinearity,
                )
                for _ in scales
            ]
        )

        fused_channels = branch_channels * (len(scales) + 1)
        self.fuse = VNLinearLeakyReLU(
            fused_channels,
            channels,
            dim=3,
            share_nonlinearity=share_nonlinearity,
        )
        expanded = max(channels, int(round(channels * ffn_ratio)))
        self.ffn = nn.Sequential(
            VNLinearLeakyReLU(
                channels,
                expanded,
                dim=3,
                share_nonlinearity=share_nonlinearity,
            ),
            VNLinear(expanded, channels),
        )

    @staticmethod
    def _sort_neighbors(points, neighbor_indices):
        relative_points = points[neighbor_indices] - points[:, None]
        squared_distance = (relative_points * relative_points).sum(dim=-1)
        order = torch.argsort(squared_distance, dim=1)
        return torch.gather(neighbor_indices, dim=1, index=order)

    def forward(self, feats, points, neighbor_indices):
        if feats.ndim != 3 or feats.shape[-1] != 3:
            raise ValueError(f'feats must have shape (N, C, 3), got {tuple(feats.shape)}')
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f'points must have shape (N, 3), got {tuple(points.shape)}')
        if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != points.shape[0]:
            raise ValueError('neighbor_indices must have shape (N, K).')
        sorted_neighbors = self._sort_neighbors(points, neighbor_indices)
        available_neighbors = sorted_neighbors.shape[1]

        identity = self.identity_branch(feats)
        scale_outputs = []
        for scale, branch in zip(self.neighbor_scales, self.scale_branches):
            effective_scale = min(scale, available_neighbors)
            scale_outputs.append(
                branch(feats, points, sorted_neighbors[:, :effective_scale])
            )

        if self.fusion_mode == 'equal' and len(scale_outputs) > 1:






            mean_scale = torch.stack(scale_outputs, dim=0).mean(dim=0)
            scale_outputs = [mean_scale for _ in scale_outputs]

        outputs = [identity] + scale_outputs
        fused = self.fuse(torch.cat(outputs, dim=1))
        return fused + self.ffn(fused)


class PAREConvFPN(nn.Module):
    pass







    def __init__(
        self,
        init_dim,
        output_dim,
        kernel_size,
        share_nonlinearity=False,
        conv_way='edge_conv',
        use_xyz=True,
        re_feature_source='official',
        coarse_context='none',
        fine_context='none',
        spsa_partial_ratio=0.5,
        spsa_qk_channels=32,
        spsa_ffn_ratio=2.0,
        mspki_neighbor_scales=(8, 16, 32),
        mspki_branch_channels=16,
        mspki_ffn_ratio=2.0,
        layer_scale_init=0.1,
        mspki_fusion='learned',
    ):
        super().__init__()
        conv_info = {'conv_way': conv_way, 'use_xyz': use_xyz}
        if re_feature_source not in ('official', 'decoder', 'encoder'):
            raise ValueError("re_feature_source must be official/decoder/encoder.")
        if coarse_context not in ('none', 'spsa', 'mspki'):
            raise ValueError('coarse_context must be none/spsa/mspki.')
        if fine_context not in ('none', 'spsa', 'mspki'):
            raise ValueError('fine_context must be none/spsa/mspki.')
        if output_dim % 3 != 0:
            raise ValueError('output_dim must be divisible by 3 for VN features.')

        self.re_feature_source = re_feature_source
        self.coarse_context = coarse_context
        self.fine_context = fine_context

        self.encoder2_1 = PARE_Conv_Block(
            1, init_dim // 3, kernel_size, share_nonlinearity=share_nonlinearity
        )
        self.encoder2_2 = PARE_Conv_Resblock(
            init_dim // 3,
            2 * init_dim // 3,
            kernel_size,
            shortcut_linear=True,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )
        self.encoder2_3 = PARE_Conv_Resblock(
            2 * init_dim // 3,
            2 * init_dim // 3,
            kernel_size,
            shortcut_linear=False,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )

        self.encoder3_1 = PARE_Conv_Resblock(
            2 * init_dim // 3,
            4 * init_dim // 3,
            kernel_size,
            shortcut_linear=True,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )
        self.encoder3_2 = PARE_Conv_Resblock(
            4 * init_dim // 3,
            4 * init_dim // 3,
            kernel_size,
            shortcut_linear=False,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )
        self.encoder3_3 = PARE_Conv_Resblock(
            4 * init_dim // 3,
            4 * init_dim // 3,
            kernel_size,
            shortcut_linear=False,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )

        self.encoder4_1 = PARE_Conv_Resblock(
            4 * init_dim // 3,
            8 * init_dim // 3,
            kernel_size,
            shortcut_linear=True,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )
        self.encoder4_2 = PARE_Conv_Resblock(
            8 * init_dim // 3,
            8 * init_dim // 3,
            kernel_size,
            shortcut_linear=False,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )
        self.encoder4_3 = PARE_Conv_Resblock(
            8 * init_dim // 3,
            8 * init_dim // 3,
            kernel_size,
            shortcut_linear=False,
            share_nonlinearity=share_nonlinearity,
            conv_info=conv_info,
        )

        encoder_channels = 8 * init_dim // 3
        decoder_channels = output_dim // 3

        self.coarse_context_module, self.coarse_context_scale = self._make_context(
            coarse_context,
            encoder_channels,
            spsa_partial_ratio,
            spsa_qk_channels,
            spsa_ffn_ratio,
            mspki_neighbor_scales,
            mspki_branch_channels,
            mspki_ffn_ratio,
            layer_scale_init,
            share_nonlinearity,
            mspki_fusion,
        )

        self.coarse_RI_head = VNLinear(encoder_channels, encoder_channels)
        self.coarse_std_feature = VNStdFeature(
            encoder_channels,
            dim=3,
            normalize_frame=True,
            share_nonlinearity=share_nonlinearity,
        )

        self.decoder3 = VNLinearLeakyReLU(
            12 * init_dim // 3,
            4 * init_dim // 3,
            dim=3,
            share_nonlinearity=share_nonlinearity,
        )
        self.decoder2 = VNLinearLeakyReLU(
            6 * init_dim // 3,
            decoder_channels,
            dim=3,
            share_nonlinearity=share_nonlinearity,
        )

        self.fine_context_module, self.fine_context_scale = self._make_context(
            fine_context,
            decoder_channels,
            spsa_partial_ratio,
            spsa_qk_channels,
            spsa_ffn_ratio,
            mspki_neighbor_scales,
            mspki_branch_channels,
            mspki_ffn_ratio,
            layer_scale_init,
            share_nonlinearity,
            mspki_fusion,
        )

        self.RI_head = VNLinear(decoder_channels, decoder_channels)
        self.RE_head = VNLinear(decoder_channels, decoder_channels)
        self.fine_std_feature = VNStdFeature(
            decoder_channels,
            dim=3,
            normalize_frame=True,
            share_nonlinearity=share_nonlinearity,
        )
        self.matching_score_proj = nn.Linear(decoder_channels * 3, 1)

    @staticmethod
    def _make_context(
        kind,
        channels,
        spsa_partial_ratio,
        spsa_qk_channels,
        spsa_ffn_ratio,
        mspki_neighbor_scales,
        mspki_branch_channels,
        mspki_ffn_ratio,
        layer_scale_init,
        share_nonlinearity,
        mspki_fusion,
    ):
        if kind == 'none':
            return None, None
        if kind == 'spsa':
            module = RotationEquivariantSPSA(
                channels,
                partial_ratio=spsa_partial_ratio,
                qk_channels=spsa_qk_channels,
                ffn_ratio=spsa_ffn_ratio,
                share_nonlinearity=share_nonlinearity,
            )
        elif kind == 'mspki':
            module = RotationEquivariantMSPKI(
                channels,
                neighbor_scales=mspki_neighbor_scales,
                branch_channels=mspki_branch_channels,
                ffn_ratio=mspki_ffn_ratio,
                share_nonlinearity=share_nonlinearity,
                fusion_mode=mspki_fusion,
            )
        else:
            raise ValueError(f'Unknown context module: {kind}')
        return module, EquivariantLayerScale(channels, layer_scale_init)

    @staticmethod
    def _apply_context(module, scale, feats, points, neighbors):
        if module is None:
            return feats
        return feats + scale(module(feats, points, neighbors))

    @property
    def spsa(self):
        if self.coarse_context == 'spsa':
            return self.coarse_context_module
        if self.fine_context == 'spsa':
            return self.fine_context_module
        return None

    @property
    def spsa_scale(self):
        if self.coarse_context == 'spsa':
            return self.coarse_context_scale
        if self.fine_context == 'spsa':
            return self.fine_context_scale
        return None

    @property
    def pki(self):
        if self.coarse_context == 'mspki':
            return self.coarse_context_module
        if self.fine_context == 'mspki':
            return self.fine_context_module
        return None

    @property
    def pki_scale(self):
        if self.coarse_context == 'mspki':
            return self.coarse_context_scale
        if self.fine_context == 'mspki':
            return self.fine_context_scale
        return None

    def forward(self, data_dict):
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = points_list[0][:, None]
        feats_s2 = self.encoder2_1(
            points_list[1], points_list[0], feats_s1, subsampling_list[0]
        )
        feats_s2 = self.encoder2_2(
            points_list[1], points_list[1], feats_s2, neighbors_list[1]
        )
        feats_s2 = self.encoder2_3(
            points_list[1], points_list[1], feats_s2, neighbors_list[1]
        )

        feats_s3 = self.encoder3_1(
            points_list[2], points_list[1], feats_s2, subsampling_list[1]
        )
        feats_s3 = self.encoder3_2(
            points_list[2], points_list[2], feats_s3, neighbors_list[2]
        )
        feats_s3 = self.encoder3_3(
            points_list[2], points_list[2], feats_s3, neighbors_list[2]
        )

        feats_s4_pre_context = self.encoder4_1(
            points_list[3], points_list[2], feats_s3, subsampling_list[2]
        )
        feats_s4_pre_context = self.encoder4_2(
            points_list[3], points_list[3], feats_s4_pre_context, neighbors_list[3]
        )
        feats_s4_pre_context = self.encoder4_3(
            points_list[3], points_list[3], feats_s4_pre_context, neighbors_list[3]
        )


        feats_s4 = self._apply_context(
            self.coarse_context_module,
            self.coarse_context_scale,
            feats_s4_pre_context,
            points_list[3],
            neighbors_list[3],
        )

        coarse_feats = self.coarse_RI_head(feats_s4)
        ri_feats_c, _ = self.coarse_std_feature(coarse_feats)
        ri_feats_c = ri_feats_c.reshape(ri_feats_c.shape[0], -1)

        up1 = upsampling_list[1]
        latent_s3 = index_select(feats_s4, up1[:, 0], dim=0)
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        up2 = upsampling_list[0]
        latent_s2 = index_select(latent_s3, up2[:, 0], dim=0)
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2_pre_context = self.decoder2(latent_s2)


        latent_s2 = self._apply_context(
            self.fine_context_module,
            self.fine_context_scale,
            latent_s2_pre_context,
            points_list[1],
            neighbors_list[1],
        )

        ri_feats = self.RI_head(latent_s2)
        decoder_re_feats_f = self.RE_head(latent_s2)
        encoder_re_feats_f = feats_s2

        ri_feats_f, _ = self.fine_std_feature(ri_feats)
        ri_feats_f = ri_feats_f.reshape(ri_feats_f.shape[0], -1)
        m_scores = self.matching_score_proj(ri_feats_f).sigmoid().squeeze(-1)



        if self.re_feature_source == 'official':
            re_feats_f = encoder_re_feats_f if not self.training else decoder_re_feats_f
        elif self.re_feature_source == 'encoder':


            re_feats_f = encoder_re_feats_f + decoder_re_feats_f.sum() * 0.0
        else:
            re_feats_f = decoder_re_feats_f

        return re_feats_f, ri_feats_f, feats_s4, ri_feats_c, m_scores

