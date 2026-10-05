
import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance
from pareconv.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        if sigma_d <= 0 or sigma_a <= 0:
            raise ValueError('sigma_d and sigma_a must be positive.')
        if angle_k < 1:
            raise ValueError('angle_k must be positive.')
        self.sigma_d = float(sigma_d)
        self.sigma_a = float(sigma_a)
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = int(angle_k)

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        pass








        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f'points must have shape (B, N, 3), got {tuple(points.shape)}')
        batch_size, num_point, _ = points.shape
        if num_point < 1:
            raise ValueError('GeometricStructureEmbedding requires at least one point.')

        dist_map = torch.sqrt(pairwise_distance(points, points).clamp_min(0.0))
        d_indices = dist_map / self.sigma_d



        k = min(self.angle_k, max(num_point - 1, 0))
        if k == 0:
            knn_indices = torch.zeros(
                (batch_size, num_point, 1), dtype=torch.long, device=points.device
            )
            k = 1
        else:
            knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)
        ref_vectors = knn_points - points.unsqueeze(2)



        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)
        angles = torch.atan2(sin_values, cos_values)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        pass













        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        pass













        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list
