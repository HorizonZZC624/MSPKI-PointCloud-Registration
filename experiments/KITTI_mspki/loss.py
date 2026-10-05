
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
        ref_feats = output_dict["ref_feats_c"]
        src_feats = output_dict["src_feats_c"]
        gt = output_dict["gt_node_corr_indices"]
        overlaps_gt = output_dict["gt_node_corr_overlaps"]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt[:, 0], gt[:, 1]] = overlaps_gt
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())
        return self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)


def _mean_or_zero(values, reference):
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
        ref_points = output_dict["ref_node_corr_knn_points"]
        src_points = output_dict["src_node_corr_knn_points"]
        ref_masks = output_dict["ref_node_corr_knn_masks"]
        src_masks = output_dict["src_node_corr_knn_masks"]
        ref_scores = output_dict["ref_node_corr_knn_scores"]
        src_scores = output_dict["src_node_corr_knn_scores"]
        matching_scores = output_dict["matching_scores"]

        transform = data_dict["transform"]
        src_points = apply_transform(src_points, transform)
        dists = pairwise_distance(ref_points, src_points)
        gt_masks = torch.logical_and(ref_masks.unsqueeze(2), src_masks.unsqueeze(1))
        gt_corr_map = torch.logical_and(
            torch.lt(dists, self.positive_radius ** 2), gt_masks
        )
        slack_rows = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_masks)
        slack_cols = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_masks)

        positive_log = matching_scores[gt_corr_map].log()
        row_log = (1 - ref_scores)[slack_rows].log()
        col_log = (1 - src_scores)[slack_cols].log()
        fine_ri_loss = -(
            _mean_or_zero(positive_log, matching_scores)
            + 0.5 * _mean_or_zero(row_log, matching_scores)
            + 0.5 * _mean_or_zero(col_log, matching_scores)
        )

        neg_map = torch.logical_and(
            torch.gt(dists, self.negative_radius ** 2), gt_masks
        )
        return fine_ri_loss, self.fine_re_loss(
            output_dict, gt_corr_map, neg_map, transform
        )

    def fine_re_loss(self, output_dict, gt_corr_map, neg_map, gt_transform):
        ref_feats = output_dict["re_ref_node_corr_knn_feats"]
        src_feats = output_dict["re_src_node_corr_knn_feats"]

        b, r, s = torch.nonzero(gt_corr_map, as_tuple=True)
        if b.numel() == 0:
            pos_loss = ref_feats.sum() * 0.0
        else:
            rf = ref_feats[b, r]
            sf = src_feats[b, s]
            sf = torch.einsum("bck,lk->bcl", sf, gt_transform[:3, :3])
            pos_loss = torch.relu(
                torch.norm(sf - rf, 2, -1) - self.positive_margin
            ).mean()

        b, r, s = torch.nonzero(neg_map, as_tuple=True)
        if b.numel() == 0:
            neg_loss = ref_feats.sum() * 0.0
        else:
            rf = ref_feats[b, r]
            sf = src_feats[b, s]
            sf = torch.einsum("bck,lk->bcl", sf, gt_transform[:3, :3])
            neg_loss = torch.relu(
                self.negative_margin - torch.norm(sf - rf, 2, -1)
            ).mean()
        return pos_loss + neg_loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.wc = cfg.loss.weight_coarse_loss
        self.wri = cfg.loss.weight_fine_ri_loss
        self.wre = cfg.loss.weight_fine_re_loss

    def forward(self, output_dict, data_dict):
        c = self.coarse_loss(output_dict)
        ri, re = self.fine_loss(output_dict, data_dict)
        return {
            "loss": self.wc * c + self.wri * ri + self.wre * re,
            "c_loss": c,
            "f_ri_loss": ri,
            "f_re_loss": re,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.rre_threshold = cfg.eval.rre_threshold
        self.rte_threshold = cfg.eval.rte_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        nr = output_dict["ref_points_c"].shape[0]
        ns = output_dict["src_points_c"].shape[0]
        gt_overlap = output_dict["gt_node_corr_overlaps"]
        gt = output_dict["gt_node_corr_indices"]
        gt = gt[torch.gt(gt_overlap, self.acceptance_overlap)]
        mapping = torch.zeros(nr, ns, device=gt.device)
        if gt.numel() > 0:
            mapping[gt[:, 0], gt[:, 1]] = 1.0
        pr = output_dict["ref_node_corr_indices"]
        ps = output_dict["src_node_corr_indices"]
        if pr.numel() == 0:
            return mapping.new_tensor(0.0)
        return mapping[pr, ps].mean()

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict["transform"]
        ref = output_dict["ref_corr_points"]
        src = output_dict["src_corr_points"]
        if src.shape[0] == 0:
            return transform.new_tensor(0.0)
        src = apply_transform(src, transform)
        return torch.lt(
            torch.linalg.norm(ref - src, dim=1), self.acceptance_radius
        ).float().mean()

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        rre, rte = isotropic_transform_error(
            data_dict["transform"], output_dict["estimated_transform"]
        )
        tr = torch.logical_and(
            torch.lt(rre, self.rre_threshold),
            torch.lt(rte, self.rte_threshold),
        ).float()
        return rre, rte, tr

    def forward(self, output_dict, data_dict, evaluate_fine=True, evaluate_registration=True):
        result = {"PIR": self.evaluate_coarse(output_dict)}
        if evaluate_fine:
            result["IR"] = self.evaluate_fine(output_dict, data_dict)
        if evaluate_registration:
            rre, rte, tr = self.evaluate_registration(output_dict, data_dict)
            result.update({"RRE": rre, "RTE": rte, "RR": tr})
        return result
