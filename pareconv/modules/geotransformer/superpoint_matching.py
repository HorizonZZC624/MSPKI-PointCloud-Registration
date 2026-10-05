import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super().__init__()
        self.num_correspondences = int(num_correspondences)
        self.dual_normalization = bool(dual_normalization)

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        device = ref_feats.device
        if ref_masks is None:
            ref_masks = torch.ones(ref_feats.shape[0], dtype=torch.bool, device=device)
        if src_masks is None:
            src_masks = torch.ones(src_feats.shape[0], dtype=torch.bool, device=device)

        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        if ref_indices.numel() == 0 or src_indices.numel() == 0:
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            empty_scores = torch.empty(0, dtype=ref_feats.dtype, device=device)
            return empty_indices, empty_indices.clone(), empty_scores

        valid_ref_feats = ref_feats[ref_indices]
        valid_src_feats = src_feats[src_indices]
        matching_scores = torch.exp(
            -pairwise_distance(valid_ref_feats, valid_src_feats, normalized=True)
        )
        if self.dual_normalization:
            ref_scores = matching_scores / matching_scores.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            src_scores = matching_scores / matching_scores.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-12)
            matching_scores = ref_scores * src_scores

        num_correspondences = min(
            self.num_correspondences, int(matching_scores.numel())
        )
        if num_correspondences == 0:
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            empty_scores = torch.empty(0, dtype=ref_feats.dtype, device=device)
            return empty_indices, empty_indices.clone(), empty_scores

        corr_scores, corr_indices = matching_scores.reshape(-1).topk(
            k=num_correspondences, largest=True
        )
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        return (
            ref_indices[ref_sel_indices],
            src_indices[src_sel_indices],
            corr_scores,
        )
