







import torch
import torch.nn as nn

from pareconv.modules.loss import WeightedCircleLoss
from pareconv.modules.ops.pairwise_distance import pairwise_distance
from pareconv.modules.ops.transformation import apply_transform
from pareconv.modules.registration.metrics import isotropic_transform_error


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())
        return self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)


def _mean_or_zero(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    pass
    if values.numel() == 0:
        return reference.sum() * 0.0
    return values.mean()


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positive_radius = cfg.fine_loss.positive_radius
        self.negative_radius = cfg.fine_loss.negative_radius
        self.positive_margin = cfg.fine_loss.positive_margin
        self.negative_margin = cfg.fine_loss.negative_margin

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        ref_node_corr_knn_scores = output_dict['ref_node_corr_knn_scores']
        src_node_corr_knn_scores = output_dict['src_node_corr_knn_scores']
        matching_scores = output_dict['matching_scores']

        transform = data_dict['transform']
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)
        gt_masks = torch.logical_and(
            ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1)
        )
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(
            torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks
        )
        slack_col_labels = torch.logical_and(
            torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks
        )

        positive_log = matching_scores[gt_corr_map].log()
        slack_row_log = (1 - ref_node_corr_knn_scores)[slack_row_labels].log()
        slack_col_log = (1 - src_node_corr_knn_scores)[slack_col_labels].log()
        fine_ri_loss = -(
            _mean_or_zero(positive_log, matching_scores)
            + 0.5 * _mean_or_zero(slack_row_log, matching_scores)
            + 0.5 * _mean_or_zero(slack_col_log, matching_scores)
        )

        neg_map = torch.gt(dists, self.negative_radius ** 2)
        neg_map = torch.logical_and(neg_map, gt_masks)
        fine_re_loss = self.fine_re_loss(output_dict, gt_corr_map, neg_map, transform)
        return fine_ri_loss, fine_re_loss

    def fine_re_loss(self, output_dict, gt_corr_map, neg_map, gt_transform):
        ref_feats = output_dict['re_ref_node_corr_knn_feats']
        src_feats = output_dict['re_src_node_corr_knn_feats']

        batch_indices, ref_indices, src_indices = torch.nonzero(gt_corr_map, as_tuple=True)
        if batch_indices.shape[0] == 0:
            return torch.tensor(0.0, device=ref_feats.device)
        ref_feats_rot = ref_feats[batch_indices, ref_indices]
        src_feats_rot = src_feats[batch_indices, src_indices]
        src_feats_rot = torch.einsum('bck, lk -> bcl', src_feats_rot, gt_transform[:3, :3])
        pos_loss = torch.relu(
            torch.norm(src_feats_rot - ref_feats_rot, 2, -1) - self.positive_margin
        ).mean()

        batch_indices, ref_indices, src_indices = torch.nonzero(neg_map, as_tuple=True)
        if batch_indices.shape[0] == 0:
            neg_loss = ref_feats.sum() * 0.0
        else:
            ref_feats_rot = ref_feats[batch_indices, ref_indices]
            src_feats_rot = src_feats[batch_indices, src_indices]
            src_feats_rot = torch.einsum('bck, lk -> bcl', src_feats_rot, gt_transform[:3, :3])
            neg_loss = torch.relu(
                self.negative_margin - torch.norm(src_feats_rot - ref_feats_rot, 2, -1)
            ).mean()
        return pos_loss + neg_loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_ri_loss = cfg.loss.weight_fine_ri_loss
        self.weight_fine_re_loss = cfg.loss.weight_fine_re_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_ri_loss, fine_re_loss = self.fine_loss(output_dict, data_dict)
        loss = (
            self.weight_coarse_loss * coarse_loss
            + self.weight_fine_ri_loss * fine_ri_loss
            + self.weight_fine_re_loss * fine_re_loss
        )
        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_ri_loss': fine_ri_loss,
            'f_re_loss': fine_re_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.feat_rre_threshold = cfg.eval.feat_rre_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c, device=gt_node_corr_indices.device)
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0
        pred_ref = output_dict['ref_node_corr_indices']
        pred_src = output_dict['src_node_corr_indices']
        if pred_ref.numel() == 0:
            return gt_node_corr_map.new_tensor(0.0)
        return gt_node_corr_map[pred_ref, pred_src].mean()

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        if src_corr_points.shape[0] == 0:
            return torch.tensor(0.0, device=transform.device)
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        return torch.lt(corr_distances, self.acceptance_radius).float().mean()

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']
        if est_transform.shape != (4, 4) or not torch.isfinite(est_transform).all():
            est_transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()
        return rre, rte, rmse, recall

    def forward(
        self,
        output_dict,
        data_dict,
        evaluate_fine: bool = True,
        evaluate_registration: bool = True,
    ):
        result = {'PIR': self.evaluate_coarse(output_dict)}
        if evaluate_fine:
            result['IR'] = self.evaluate_fine(output_dict, data_dict)
        if evaluate_registration:
            rre, rte, rmse, rr = self.evaluate_registration(output_dict, data_dict)
            result.update({'RRE': rre, 'RTE': rte, 'RMSE': rmse, 'RR': rr})
        return result
