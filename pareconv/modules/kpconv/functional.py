import torch

from pareconv.modules.ops import index_select


def nearest_upsample(x, upsample_indices):
    pass











    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    x = index_select(x, upsample_indices[:, 0], dim=0)
    return x


def knn_interpolate(s_feats, q_points, s_points, neighbor_indices, k, eps=1e-8):
    pass














    s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)
    s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)
    knn_indices = neighbor_indices[:, :k].contiguous()
    knn_points = index_select(s_points, knn_indices, dim=0)
    knn_feats = index_select(s_feats, knn_indices, dim=0)
    knn_sq_distances = (q_points.unsqueeze(1) - knn_points).pow(2).sum(dim=-1)
    knn_masks = torch.ne(knn_indices, s_points.shape[0] - 1).float()
    knn_weights = knn_masks / (knn_sq_distances + eps)
    knn_weights = knn_weights / (knn_weights.sum(dim=1, keepdim=True) + eps)
    q_feats = (knn_feats * knn_weights.unsqueeze(-1)).sum(dim=1)
    return q_feats


def maxpool(x, neighbor_indices):
    pass








    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats = neighbor_feats.max(1)[0]
    return pooled_feats


def global_avgpool(x, batch_lengths):
    pass









    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):

        averaged_features.append(torch.mean(x[i0 : i0 + length], dim=0))

        i0 += length

    x = torch.stack(averaged_features)
    return x
