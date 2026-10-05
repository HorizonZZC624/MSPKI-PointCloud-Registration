from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pareconv.modules.dual_matching import PointDualMatching
from pareconv.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
)
from pareconv.modules.ops import GPUStackModePreprocessor, index_select, point_to_node_partition
from pareconv.modules.registration import HypothesisProposer, get_node_correspondences

from backbone import PAREConvFPN


class PARE_Net(nn.Module):
    pass







    def __init__(self, cfg):
        super().__init__()
        self.num_points_in_patch = int(cfg.model.num_points_in_patch)
        self.matching_radius = float(cfg.model.ground_truth_matching_radius)
        self.estimate_transform_during_training = bool(
            cfg.model.estimate_transform_during_training
        )
        self.re_feature_source = cfg.fine_matching.re_feature_source

        self.preprocessor = GPUStackModePreprocessor(
            num_stages=cfg.backbone.num_stages,
            voxel_size=cfg.backbone.init_voxel_size,
            num_neighbors=cfg.backbone.num_neighbors,
            subsample_ratio=cfg.backbone.subsample_ratio,
            query_chunk_size=cfg.preprocess.query_chunk_size,
            support_chunk_size=cfg.preprocess.support_chunk_size,
            require_cuda=cfg.preprocess.require_cuda,
            skip_self_neighbor_stages=(0,),
        )

        self.backbone = PAREConvFPN(
            cfg.backbone.init_dim,
            cfg.backbone.output_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.share_nonlinearity,
            cfg.backbone.conv_way,
            cfg.backbone.use_xyz,
            re_feature_source=cfg.fine_matching.re_feature_source,
            coarse_context=cfg.backbone.coarse_context,
            fine_context=cfg.backbone.fine_context,
            spsa_partial_ratio=cfg.backbone.spsa_partial_ratio,
            spsa_qk_channels=cfg.backbone.spsa_qk_channels,
            spsa_ffn_ratio=cfg.backbone.spsa_ffn_ratio,
            mspki_neighbor_scales=cfg.backbone.mspki_neighbor_scales,
            mspki_branch_channels=cfg.backbone.mspki_branch_channels,
            mspki_ffn_ratio=cfg.backbone.mspki_ffn_ratio,
            layer_scale_init=cfg.backbone.layer_scale_init,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets,
            cfg.coarse_matching.overlap_threshold,
        )
        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences,
            cfg.coarse_matching.dual_normalization,
        )
        self.fine_matching = HypothesisProposer(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            num_hypotheses=cfg.fine_matching.num_hypotheses,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )
        self.point_matching = PointDualMatching(
            dim=cfg.backbone.output_dim // 3 * 3
        )

    @staticmethod
    def _assert_batch_size_one(data_dict: Dict) -> None:
        batch_size = int(data_dict.get('batch_size', 1))
        if batch_size != 1:
            raise NotImplementedError('PARE-Net stack split assumes batch_size=1.')

    def _empty_registration(self, reference: torch.Tensor):



        return {
            're_ref_corr_feats': reference.new_empty((0, 1, 3)),
            're_src_corr_feats': reference.new_empty((0, 1, 3)),
            'hypotheses': reference.new_empty((0, 4, 4)),
            'hypothesis_sources': torch.empty((0,), dtype=torch.long, device=reference.device),
            'ref_corr_points': reference.new_empty((0, 3)),
            'src_corr_points': reference.new_empty((0, 3)),
            'corr_scores': reference.new_empty((0,)),
            'estimated_transform': torch.eye(4, dtype=reference.dtype, device=reference.device),
        }

    def _current_re_source_id(self) -> int:

        if self.re_feature_source == 'encoder':
            return 0
        if self.re_feature_source == 'official' and not self.training:
            return 0
        return 1

    def forward(
        self,
        data_dict: Dict,
        compute_gt: Optional[bool] = None,
        estimate_transform: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        self._assert_batch_size_one(data_dict)
        data_dict = self.preprocessor(data_dict)
        self._assert_batch_size_one(data_dict)

        has_transform = isinstance(data_dict.get('transform'), torch.Tensor)
        if compute_gt is None:
            compute_gt = has_transform
        if compute_gt and not has_transform:
            raise KeyError('compute_gt=True requires data_dict["transform"].')
        if self.training and not compute_gt:
            raise RuntimeError('Training requires the ground-truth transform.')
        if estimate_transform is None:
            estimate_transform = (
                self.estimate_transform_during_training if self.training else True
            )

        output_dict: Dict[str, torch.Tensor] = {}
        transform = data_dict['transform'].detach() if has_transform else None

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict.update(
            {
                'ref_points_c': ref_points_c,
                'src_points_c': src_points_c,
                'ref_points_f': ref_points_f,
                'src_points_f': src_points_f,
                'ref_points': ref_points,
                'src_points': src_points,
            }
        )


        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )
        output_dict['ref_node_knn_indices'] = ref_node_knn_indices
        output_dict['src_node_knn_indices'] = src_node_knn_indices

        ref_padded_points_f = torch.cat(
            [ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0
        )
        src_padded_points_f = torch.cat(
            [src_points_f, torch.zeros_like(src_points_f[:1])], dim=0
        )
        ref_node_knn_points = index_select(
            ref_padded_points_f, ref_node_knn_indices, dim=0
        )
        src_node_knn_points = index_select(
            src_padded_points_f, src_node_knn_indices, dim=0
        )

        gt_node_corr_indices = None
        gt_node_corr_overlaps = None
        if compute_gt:
            gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
                ref_points_c,
                src_points_c,
                ref_node_knn_points,
                src_node_knn_points,
                transform,
                self.matching_radius,
                ref_masks=ref_node_masks,
                src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks,
                src_knn_masks=src_node_knn_masks,
            )
            output_dict['gt_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps


        re_feats_f, feats_f, re_feats_c, feats_c, m_scores = self.backbone(data_dict)


        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        output_dict['ref_feats_c_re'] = re_feats_c[:ref_length_c]
        output_dict['src_feats_c_re'] = re_feats_c[ref_length_c:]

        ref_feats_c, src_feats_c, _ = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm


        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        m_ref_scores = m_scores[:ref_length_f]
        m_src_scores = m_scores[ref_length_f:]
        re_ref_feats_f = re_feats_f[:ref_length_f]
        re_src_feats_f = re_feats_f[ref_length_f:]
        output_dict.update(
            {
                'm_ref_scores': m_ref_scores,
                'm_src_scores': m_src_scores,
                'ref_feats_f': ref_feats_f,
                'src_feats_f': src_feats_f,
                're_ref_feats_f': re_ref_feats_f,
                're_src_feats_f': re_src_feats_f,
            }
        )



        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            output_dict['node_corr_scores'] = node_corr_scores

            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )


        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(
            ref_padded_feats_f, ref_node_corr_knn_indices, dim=0
        )
        src_node_corr_knn_feats = index_select(
            src_padded_feats_f, src_node_corr_knn_indices, dim=0
        )

        m_ref_padded_scores = torch.cat([m_ref_scores, torch.zeros_like(m_ref_scores[:1])], dim=0)
        m_src_padded_scores = torch.cat([m_src_scores, torch.zeros_like(m_src_scores[:1])], dim=0)
        ref_node_corr_knn_scores = index_select(
            m_ref_padded_scores, ref_node_corr_knn_indices, dim=0
        )
        src_node_corr_knn_scores = index_select(
            m_src_padded_scores, src_node_corr_knn_indices, dim=0
        )

        re_ref_padded_feats_f = torch.cat(
            [re_ref_feats_f, torch.zeros_like(re_ref_feats_f[:1])], dim=0
        )
        re_src_padded_feats_f = torch.cat(
            [re_src_feats_f, torch.zeros_like(re_src_feats_f[:1])], dim=0
        )
        re_ref_node_corr_knn_feats = index_select(
            re_ref_padded_feats_f, ref_node_corr_knn_indices, dim=0
        )
        re_src_node_corr_knn_feats = index_select(
            re_src_padded_feats_f, src_node_corr_knn_indices, dim=0
        )

        output_dict.update(
            {
                'ref_node_corr_knn_points': ref_node_corr_knn_points,
                'src_node_corr_knn_points': src_node_corr_knn_points,
                'ref_node_corr_knn_masks': ref_node_corr_knn_masks,
                'src_node_corr_knn_masks': src_node_corr_knn_masks,
                'ref_node_corr_knn_scores': ref_node_corr_knn_scores,
                'src_node_corr_knn_scores': src_node_corr_knn_scores,
                're_ref_node_corr_knn_feats': re_ref_node_corr_knn_feats,
                're_src_node_corr_knn_feats': re_src_node_corr_knn_feats,
            }
        )


        matching_scores = self.point_matching(
            ref_node_corr_knn_feats,
            src_node_corr_knn_feats,
            ref_node_corr_knn_scores,
            src_node_corr_knn_scores,
            ref_node_corr_knn_masks,
            src_node_corr_knn_masks,
        )
        output_dict['matching_scores'] = matching_scores



        registration_attempted = bool(estimate_transform)
        registration_succeeded = False
        registration = self._empty_registration(ref_points)
        if estimate_transform and matching_scores.shape[0] > 0:
            with torch.no_grad():
                (
                    ref_corr_points,
                    src_corr_points,
                    corr_scores,
                    estimated_transform,
                    hypotheses,
                    re_ref_corr_feats,
                    re_src_corr_feats,
                ) = self.fine_matching(
                    ref_node_corr_knn_points,
                    src_node_corr_knn_points,
                    re_ref_node_corr_knn_feats,
                    re_src_node_corr_knn_feats,
                    ref_node_corr_knn_masks,
                    src_node_corr_knn_masks,
                    matching_scores,
                )
            source_id = self._current_re_source_id()
            hypothesis_sources = torch.full(
                (hypotheses.shape[0],),
                source_id,
                dtype=torch.long,
                device=hypotheses.device,
            )
            registration = {
                're_ref_corr_feats': re_ref_corr_feats,
                're_src_corr_feats': re_src_corr_feats,
                'hypotheses': hypotheses,
                'hypothesis_sources': hypothesis_sources,
                'ref_corr_points': ref_corr_points,
                'src_corr_points': src_corr_points,
                'corr_scores': corr_scores,
                'estimated_transform': estimated_transform,
            }
            registration_succeeded = bool(
                estimated_transform.shape == (4, 4)
                and torch.isfinite(estimated_transform).all().item()
                and ref_corr_points.shape[0] > 0
            )

        output_dict.update(registration)
        output_dict['registration_attempted'] = torch.tensor(
            registration_attempted, dtype=torch.bool, device=ref_points.device
        )
        output_dict['registration_succeeded'] = torch.tensor(
            registration_succeeded, dtype=torch.bool, device=ref_points.device
        )
        if transform is not None:
            output_dict['transform'] = transform
        return output_dict


def create_model(config):
    return PARE_Net(config)


if __name__ == '__main__':
    from config import make_cfg

    model = create_model(make_cfg())
    print(model)
