import torch
import torch.nn as nn


class PointMatching(nn.Module):
    def __init__(
        self,
        k: int,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        remove_duplicate: bool = False,
    ):
        super().__init__()
        if k < 1:
            raise ValueError('k must be positive.')
        self.k = int(k)
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.remove_duplicate = remove_duplicate

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        if score_mat.ndim != 3:
            raise ValueError('score_mat must have shape (B, N, M).')
        mask_mat = ref_knn_masks.unsqueeze(2) & src_knn_masks.unsqueeze(1)
        batch_size, ref_length, src_length = score_mat.shape
        if ref_length == 0 or src_length == 0:
            return torch.zeros_like(mask_mat)
        device = score_mat.device
        batch_indices = torch.arange(batch_size, device=device)
        ref_k = min(self.k, src_length)
        src_k = min(self.k, ref_length)
        masked = torch.nan_to_num(score_mat, nan=0.0).masked_fill(
            ~mask_mat, torch.finfo(score_mat.dtype).min
        )

        ref_topk_scores, ref_topk_indices = masked.topk(k=ref_k, dim=2)
        ref_batch_indices = batch_indices[:, None, None].expand(-1, ref_length, ref_k)
        ref_indices = torch.arange(ref_length, device=device)[None, :, None].expand(
            batch_size, -1, ref_k
        )
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = ref_score_mat > self.confidence_threshold

        src_topk_scores, src_topk_indices = masked.topk(k=src_k, dim=1)
        src_batch_indices = batch_indices[:, None, None].expand(-1, src_k, src_length)
        src_indices = torch.arange(src_length, device=device)[None, None, :].expand(
            batch_size, src_k, -1
        )
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = src_score_mat > self.confidence_threshold

        corr_mat = (
            ref_corr_mat & src_corr_mat
            if self.mutual
            else ref_corr_mat | src_corr_mat
        )
        return corr_mat & mask_mat

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        ref_knn_indices,
        src_knn_indices,
        score_mat,
        global_scores,
    ):
        score_mat = torch.exp(torch.nan_to_num(score_mat, nan=-1e4))
        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]

        corr_mat = self.compute_correspondence_matrix(
            score_mat, ref_knn_masks, src_knn_masks
        )
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.to(score_mat.dtype)

        batch_indices, ref_indices, src_indices = torch.nonzero(
            corr_mat, as_tuple=True
        )
        ref_corr_indices = ref_knn_indices[batch_indices, ref_indices]
        src_corr_indices = src_knn_indices[batch_indices, src_indices]
        ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        src_corr_points = src_knn_points[batch_indices, src_indices]
        corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        if self.remove_duplicate and ref_corr_indices.numel() > 0:
            pair_keys = ref_corr_indices * (src_knn_indices.max() + 2) + src_corr_indices
            _, inverse = torch.unique(pair_keys, return_inverse=True)

            keep = []
            for group in range(int(inverse.max().item()) + 1):
                members = torch.nonzero(inverse == group, as_tuple=True)[0]
                keep.append(members[corr_scores[members].argmax()])
            keep = torch.stack(keep)
            ref_corr_indices = ref_corr_indices[keep]
            src_corr_indices = src_corr_indices[keep]
            ref_corr_points = ref_corr_points[keep]
            src_corr_points = src_corr_points[keep]
            corr_scores = corr_scores[keep]

        return (
            ref_corr_points,
            src_corr_points,
            ref_corr_indices,
            src_corr_indices,
            corr_scores,
        )
