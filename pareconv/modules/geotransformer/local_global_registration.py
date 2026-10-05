from typing import Optional

import torch
import torch.nn as nn

from pareconv.modules.ops import apply_transform
from pareconv.modules.registration import WeightedProcrustes


class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        super().__init__()
        if k < 1 or correspondence_threshold < 1 or num_refinement_steps < 1:
            raise ValueError('k, correspondence_threshold and refinement steps must be positive.')
        self.k = int(k)
        self.acceptance_radius = float(acceptance_radius)
        self.mutual = mutual
        self.confidence_threshold = float(confidence_threshold)
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = int(correspondence_threshold)
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = int(num_refinement_steps)
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
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

        ref_scores, ref_topk = masked.topk(k=ref_k, dim=2)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[
            batch_indices[:, None, None].expand(-1, ref_length, ref_k),
            torch.arange(ref_length, device=device)[None, :, None].expand(batch_size, -1, ref_k),
            ref_topk,
        ] = ref_scores

        src_scores, src_topk = masked.topk(k=src_k, dim=1)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[
            batch_indices[:, None, None].expand(-1, src_k, src_length),
            src_topk,
            torch.arange(src_length, device=device)[None, None, :].expand(batch_size, src_k, -1),
        ] = src_scores

        ref_corr = ref_score_mat > self.confidence_threshold
        src_corr = src_score_mat > self.confidence_threshold
        corr = ref_corr & src_corr if self.mutual else ref_corr | src_corr
        return corr & mask_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
        if not chunks:
            device = ref_corr_points.device
            dtype = ref_corr_points.dtype
            return (
                torch.empty((0, 0, 3), device=device, dtype=dtype),
                torch.empty((0, 0, 3), device=device, dtype=dtype),
                torch.empty((0, 0), device=device, dtype=corr_scores.dtype),
            )
        device = ref_corr_points.device
        max_corr = max(y - x for x, y in chunks)
        batch_size = len(chunks)
        batch_ref = ref_corr_points.new_zeros((batch_size, max_corr, 3))
        batch_src = src_corr_points.new_zeros((batch_size, max_corr, 3))
        batch_scores = corr_scores.new_zeros((batch_size, max_corr))
        for batch_index, (start, end) in enumerate(chunks):
            length = end - start
            batch_ref[batch_index, :length] = ref_corr_points[start:end]
            batch_src[batch_index, :length] = src_corr_points[start:end]
            batch_scores[batch_index, :length] = corr_scores[start:end]
        return batch_ref, batch_src, batch_scores

    def recompute_correspondence_scores(
        self, ref_corr_points, src_corr_points, corr_scores, estimated_transform
    ):
        aligned_src = apply_transform(src_corr_points, estimated_transform)
        residuals = torch.linalg.norm(ref_corr_points - aligned_src, dim=1)
        return corr_scores * (residuals < self.acceptance_radius).to(corr_scores.dtype)

    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        if batch_indices.numel() == 0:
            device = ref_knn_points.device
            dtype = ref_knn_points.dtype
            return (
                torch.empty((0, 3), device=device, dtype=dtype),
                torch.empty((0, 3), device=device, dtype=dtype),
                torch.empty((0,), device=device, dtype=score_mat.dtype),
                torch.eye(4, device=device, dtype=dtype),
            )

        global_ref = ref_knn_points[batch_indices, ref_indices]
        global_src = src_knn_points[batch_indices, src_indices]
        global_scores = score_mat[batch_indices, ref_indices, src_indices]

        if self.correspondence_limit is not None and global_scores.numel() > self.correspondence_limit:
            corr_scores, selected = global_scores.topk(self.correspondence_limit)
            ref_corr = global_ref[selected]
            src_corr = global_src[selected]
        else:
            ref_corr, src_corr, corr_scores = global_ref, global_src, global_scores

        boundaries = torch.nonzero(batch_indices[1:] != batch_indices[:-1], as_tuple=True)[0] + 1
        boundaries = [0] + boundaries.cpu().tolist() + [batch_indices.shape[0]]
        chunks = [
            (x, y)
            for x, y in zip(boundaries[:-1], boundaries[1:])
            if y - x >= self.correspondence_threshold
        ]

        estimated_transform = torch.eye(4, device=ref_corr.device, dtype=ref_corr.dtype)
        current_scores = corr_scores
        if chunks:
            batch_ref, batch_src, batch_scores = self.convert_to_batch(
                global_ref, global_src, global_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src, batch_ref, batch_scores)
            aligned = apply_transform(src_corr.unsqueeze(0), batch_transforms)
            residuals = torch.linalg.norm(ref_corr.unsqueeze(0) - aligned, dim=2)
            inliers = residuals < self.acceptance_radius
            best = inliers.sum(dim=1).argmax()
            estimated_transform = batch_transforms[best]
            current_scores = corr_scores * inliers[best].to(corr_scores.dtype)

        for _ in range(self.num_refinement_steps):
            if int(torch.count_nonzero(current_scores > 0).item()) < 3:
                break
            estimated_transform = self.procrustes(src_corr, ref_corr, current_scores)
            current_scores = self.recompute_correspondence_scores(
                ref_corr, src_corr, corr_scores, estimated_transform
            )

        return global_ref, global_src, global_scores, estimated_transform

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
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
        return self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat
        )
