import warnings

import torch

from pareconv.modules.ops.pairwise_distance import pairwise_distance


def _validate_points_nodes(points, nodes):
    if points.ndim != 2 or nodes.ndim != 2:
        raise ValueError('points and nodes must be rank-2 tensors.')
    if points.shape[1] != nodes.shape[1]:
        raise ValueError('points and nodes must have the same coordinate dimension.')
    if points.device != nodes.device:
        raise ValueError('points and nodes must be on the same device.')
    if points.shape[0] == 0:
        raise ValueError('points must not be empty.')
    if nodes.shape[0] == 0:
        raise ValueError('nodes must not be empty.')


def get_point_to_node_indices(
    points: torch.Tensor,
    nodes: torch.Tensor,
    return_counts: bool = False,
):
    pass
    _validate_points_nodes(points, nodes)
    sq_dist_mat = pairwise_distance(points, nodes)
    indices = sq_dist_mat.argmin(dim=1)
    if not return_counts:
        return indices

    unique_indices, unique_counts = torch.unique(indices, return_counts=True)
    node_sizes = torch.zeros(
        nodes.shape[0], dtype=torch.long, device=nodes.device
    )
    node_sizes[unique_indices] = unique_counts
    return indices, node_sizes


@torch.no_grad()
def knn_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    k: int,
    return_distance: bool = False,
):
    pass
    _validate_points_nodes(points, nodes)
    if k < 1:
        raise ValueError('k must be positive.')
    requested_k = int(k)
    effective_k = min(requested_k, int(points.shape[0]))
    sq_dist_mat = pairwise_distance(nodes, points)
    knn_sq_distances, knn_indices = sq_dist_mat.topk(
        dim=1, k=effective_k, largest=False, sorted=True
    )
    if effective_k < requested_k:
        index_padding = knn_indices[:, -1:].expand(-1, requested_k - effective_k)
        distance_padding = torch.full(
            (nodes.shape[0], requested_k - effective_k),
            float('inf'),
            dtype=knn_sq_distances.dtype,
            device=knn_sq_distances.device,
        )
        knn_indices = torch.cat([knn_indices, index_padding], dim=1)
        knn_sq_distances = torch.cat(
            [knn_sq_distances, distance_padding], dim=1
        )
    if return_distance:
        return torch.sqrt(knn_sq_distances), knn_indices
    return knn_indices


@torch.no_grad()
def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    pass




    _validate_points_nodes(points, nodes)
    if point_limit < 1:
        raise ValueError('point_limit must be positive.')

    sq_dist_mat = pairwise_distance(nodes, points)
    point_to_node = sq_dist_mat.argmin(dim=0)

    node_masks = torch.zeros(
        nodes.shape[0], dtype=torch.bool, device=nodes.device
    )
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)
    point_indices = torch.arange(points.shape[0], device=points.device)
    matching_masks[point_to_node, point_indices] = True
    masked_distances = sq_dist_mat.masked_fill(~matching_masks, float('inf'))

    effective_k = min(int(point_limit), int(points.shape[0]))
    node_knn_indices = masked_distances.topk(
        k=effective_k, dim=1, largest=False, sorted=True
    ).indices
    node_indices = torch.arange(nodes.shape[0], device=nodes.device).unsqueeze(1)
    node_knn_masks = point_to_node[node_knn_indices] == node_indices
    node_knn_indices = node_knn_indices.masked_fill(
        ~node_knn_masks, points.shape[0]
    )

    if effective_k < point_limit:
        padding_size = int(point_limit) - effective_k
        index_padding = torch.full(
            (nodes.shape[0], padding_size),
            points.shape[0],
            dtype=torch.long,
            device=points.device,
        )
        mask_padding = torch.zeros(
            (nodes.shape[0], padding_size),
            dtype=torch.bool,
            device=points.device,
        )
        node_knn_indices = torch.cat([node_knn_indices, index_padding], dim=1)
        node_knn_masks = torch.cat([node_knn_masks, mask_padding], dim=1)

    if return_count:
        unique_indices, unique_counts = torch.unique(
            point_to_node, return_counts=True
        )
        node_sizes = torch.zeros(
            nodes.shape[0], dtype=torch.long, device=nodes.device
        )
        node_sizes[unique_indices] = unique_counts
        return (
            point_to_node,
            node_sizes,
            node_masks,
            node_knn_indices,
            node_knn_masks,
        )
    return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def ball_query_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
):
    if radius <= 0:
        raise ValueError('radius must be positive.')
    node_knn_distances, node_knn_indices = knn_partition(
        points, nodes, point_limit, return_distance=True
    )
    node_knn_masks = node_knn_distances < radius
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])
    node_knn_indices = torch.where(
        node_knn_masks, node_knn_indices, sentinel_indices
    )
    node_masks = node_knn_masks.any(dim=-1)
    if return_count:
        node_sizes = node_knn_masks.sum(1)
        return node_masks, node_knn_indices, node_knn_masks, node_sizes
    return node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def point_to_node_partition_bug(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    pass
    warnings.warn(
        'point_to_node_partition_bug is deprecated; using the corrected '
        'point_to_node_partition implementation.',
        DeprecationWarning,
        stacklevel=2,
    )
    return point_to_node_partition(
        points,
        nodes,
        point_limit,
        return_count=return_count,
    )
