











from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def _as_length_list(lengths: torch.Tensor) -> List[int]:
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be one-dimensional, got {tuple(lengths.shape)}")



    return [int(value) for value in lengths.detach().cpu().tolist()]


def _validate_stacked_points(points: torch.Tensor, lengths: torch.Tensor) -> List[int]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise TypeError(f"points must be floating point, got {points.dtype}")
    length_list = _as_length_list(lengths)
    if sum(length_list) != int(points.shape[0]):
        raise ValueError(
            f"sum(lengths)={sum(length_list)} does not match points.shape[0]={points.shape[0]}"
        )
    if any(length < 0 for length in length_list):
        raise ValueError("lengths must be non-negative")
    return length_list


@torch.no_grad()
def voxel_grid_subsample_stacked(
    points: torch.Tensor,
    lengths: torch.Tensor,
    voxel_size: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass















    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    length_list = _validate_stacked_points(points, lengths)

    sampled_clouds: List[torch.Tensor] = []
    sampled_lengths: List[int] = []
    start = 0
    for length in length_list:
        end = start + length
        cloud = points[start:end]
        if length == 0:
            sampled = cloud.clone()
        else:
            origin = cloud.amin(dim=0, keepdim=True)
            voxel_coordinates = torch.floor((cloud - origin) / float(voxel_size)).to(torch.int64)
            unique_voxels, inverse = torch.unique(
                voxel_coordinates,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            num_voxels = int(unique_voxels.shape[0])

            sampled = torch.zeros(
                (num_voxels, 3),
                dtype=cloud.dtype,
                device=cloud.device,
            )
            sampled.index_add_(0, inverse, cloud)
            counts = torch.bincount(inverse, minlength=num_voxels).to(cloud.dtype)
            sampled = sampled / counts.clamp_min_(1).unsqueeze(1)

        sampled_clouds.append(sampled)
        sampled_lengths.append(int(sampled.shape[0]))
        start = end

    if sampled_clouds:
        output_points = torch.cat(sampled_clouds, dim=0)
    else:
        output_points = points.new_empty((0, 3))
    output_lengths = torch.tensor(sampled_lengths, dtype=torch.long, device=lengths.device)
    return output_points.contiguous(), output_lengths.contiguous()


def _squared_distance_block(
    query: torch.Tensor,
    support: torch.Tensor,
    query_norm: torch.Tensor | None = None,
    support_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    pass




    if query_norm is None:
        query_norm = (query * query).sum(dim=1, keepdim=True)
    if support_norm is None:
        support_norm = (support * support).sum(dim=1).unsqueeze(0)
    distances = query_norm + support_norm - 2.0 * torch.matmul(
        query, support.transpose(0, 1)
    )
    return distances.clamp_min_(0.0)


@torch.no_grad()
def exact_knn_chunked(
    query: torch.Tensor,
    support: torch.Tensor,
    num_neighbors: int,
    query_chunk_size: int = 1024,
    support_chunk_size: int = 4096,
) -> torch.Tensor:
    pass





    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError(f"query must have shape (N, 3), got {tuple(query.shape)}")
    if support.ndim != 2 or support.shape[1] != 3:
        raise ValueError(f"support must have shape (M, 3), got {tuple(support.shape)}")
    if query.device != support.device:
        raise ValueError("query and support must be on the same device")
    if num_neighbors <= 0:
        raise ValueError(f"num_neighbors must be positive, got {num_neighbors}")
    if query_chunk_size <= 0 or support_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if support.shape[0] == 0:
        raise ValueError("support point cloud must not be empty")
    if query.shape[0] == 0:
        return torch.empty(
            (0, int(num_neighbors)),
            dtype=torch.long,
            device=query.device,
        )

    requested_k = int(num_neighbors)
    effective_k = min(requested_k, int(support.shape[0]))
    output_indices = torch.empty(
        (query.shape[0], effective_k),
        dtype=torch.long,
        device=query.device,
    )



    support_squared_norms = (support * support).sum(dim=1)

    for query_start in range(0, int(query.shape[0]), int(query_chunk_size)):
        query_end = min(query_start + int(query_chunk_size), int(query.shape[0]))
        query_block = query[query_start:query_end]
        query_squared_norms = (query_block * query_block).sum(
            dim=1, keepdim=True
        )

        best_distances = torch.full(
            (query_block.shape[0], effective_k),
            float("inf"),
            dtype=query_block.dtype,
            device=query_block.device,
        )
        best_indices = torch.zeros(
            (query_block.shape[0], effective_k),
            dtype=torch.long,
            device=query_block.device,
        )

        for support_start in range(0, int(support.shape[0]), int(support_chunk_size)):
            support_end = min(support_start + int(support_chunk_size), int(support.shape[0]))
            support_block = support[support_start:support_end]
            support_norm = support_squared_norms[
                support_start:support_end
            ].unsqueeze(0)
            distances = _squared_distance_block(
                query_block,
                support_block,
                query_norm=query_squared_norms,
                support_norm=support_norm,
            )

            local_k = min(effective_k, int(support_block.shape[0]))
            local_distances, local_indices = torch.topk(
                distances,
                k=local_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            local_indices = local_indices + support_start

            candidate_distances = torch.cat((best_distances, local_distances), dim=1)
            candidate_indices = torch.cat((best_indices, local_indices), dim=1)
            best_distances, positions = torch.topk(
                candidate_distances,
                k=effective_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = torch.gather(candidate_indices, dim=1, index=positions)

        output_indices[query_start:query_end] = best_indices

    if effective_k < requested_k:
        padding = output_indices[:, -1:].expand(-1, requested_k - effective_k)
        output_indices = torch.cat((output_indices, padding), dim=1)
    return output_indices.contiguous()


@torch.no_grad()
def stacked_exact_knn(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    query_lengths: torch.Tensor,
    support_lengths: torch.Tensor,
    num_neighbors: int,
    query_chunk_size: int = 1024,
    support_chunk_size: int = 4096,
) -> torch.Tensor:
    pass





    query_length_list = _validate_stacked_points(query_points, query_lengths)
    support_length_list = _validate_stacked_points(support_points, support_lengths)
    if len(query_length_list) != len(support_length_list):
        raise ValueError("query_lengths and support_lengths must have the same number of clouds")
    if query_points.device != support_points.device:
        raise ValueError("query_points and support_points must be on the same device")

    batches: List[torch.Tensor] = []
    query_start = 0
    support_start = 0
    for query_length, support_length in zip(query_length_list, support_length_list):
        query_end = query_start + query_length
        support_end = support_start + support_length
        local_indices = exact_knn_chunked(
            query_points[query_start:query_end],
            support_points[support_start:support_end],
            num_neighbors=num_neighbors,
            query_chunk_size=query_chunk_size,
            support_chunk_size=support_chunk_size,
        )
        batches.append(local_indices + support_start)
        query_start = query_end
        support_start = support_end

    if batches:
        return torch.cat(batches, dim=0).contiguous()
    return torch.empty(
        (0, int(num_neighbors)),
        dtype=torch.long,
        device=query_points.device,
    )


@torch.no_grad()
def precompute_stack_mode_gpu(
    points: torch.Tensor,
    lengths: torch.Tensor,
    num_stages: int,
    voxel_size: float,
    num_neighbors: Sequence[int],
    subsample_ratio: float,
    query_chunk_size: int = 1024,
    support_chunk_size: int = 4096,
    skip_self_neighbor_stages: Sequence[int] = (),
) -> Dict[str, List[torch.Tensor]]:
    pass






    if num_stages != len(num_neighbors):
        raise ValueError("num_stages must equal len(num_neighbors)")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    skipped_stages = {int(stage) for stage in skip_self_neighbor_stages}
    invalid_stages = [
        stage for stage in skipped_stages if stage < 0 or stage >= num_stages
    ]
    if invalid_stages:
        raise ValueError(
            f"Invalid skipped self-neighbor stages: {sorted(invalid_stages)}"
        )

    points_list: List[torch.Tensor] = []
    lengths_list: List[torch.Tensor] = []
    current_points = points
    current_lengths = lengths
    current_voxel_size = float(voxel_size)

    for stage in range(num_stages):
        if stage > 0:
            current_points, current_lengths = voxel_grid_subsample_stacked(
                current_points,
                current_lengths,
                voxel_size=current_voxel_size,
            )
        points_list.append(current_points)
        lengths_list.append(current_lengths)
        current_voxel_size *= float(subsample_ratio)

    neighbors_list: List[torch.Tensor] = []
    subsampling_list: List[torch.Tensor] = []
    upsampling_list: List[torch.Tensor] = []

    for stage in range(num_stages):
        current_points = points_list[stage]
        current_lengths = lengths_list[stage]
        if stage in skipped_stages:
            neighbors_list.append(
                torch.empty(
                    (current_points.shape[0], 0),
                    dtype=torch.long,
                    device=current_points.device,
                )
            )
        else:
            neighbors_list.append(
                stacked_exact_knn(
                    current_points,
                    current_points,
                    current_lengths,
                    current_lengths,
                    num_neighbors=int(num_neighbors[stage]),
                    query_chunk_size=query_chunk_size,
                    support_chunk_size=support_chunk_size,
                )
            )

        if stage < num_stages - 1:
            subsampled_points = points_list[stage + 1]
            subsampled_lengths = lengths_list[stage + 1]
            subsampling_list.append(
                stacked_exact_knn(
                    subsampled_points,
                    current_points,
                    subsampled_lengths,
                    current_lengths,
                    num_neighbors=int(num_neighbors[stage]),
                    query_chunk_size=query_chunk_size,
                    support_chunk_size=support_chunk_size,
                )
            )



            if stage > 0:
                upsampling_list.append(
                    stacked_exact_knn(
                        current_points,
                        subsampled_points,
                        current_lengths,
                        subsampled_lengths,
                        num_neighbors=1,
                        query_chunk_size=query_chunk_size,
                        support_chunk_size=support_chunk_size,
                    )
                )

    return {
        "points": points_list,
        "lengths": lengths_list,
        "neighbors": neighbors_list,
        "subsampling": subsampling_list,
        "upsampling": upsampling_list,
    }


class GPUStackModePreprocessor:
    pass





    def __init__(
        self,
        num_stages: int,
        voxel_size: float,
        num_neighbors: Sequence[int],
        subsample_ratio: float,
        query_chunk_size: int = 1024,
        support_chunk_size: int = 4096,
        require_cuda: bool = True,
        skip_self_neighbor_stages: Sequence[int] = (),
    ) -> None:
        self.num_stages = int(num_stages)
        self.voxel_size = float(voxel_size)
        self.num_neighbors = tuple(int(value) for value in num_neighbors)
        self.subsample_ratio = float(subsample_ratio)
        self.query_chunk_size = int(query_chunk_size)
        self.support_chunk_size = int(support_chunk_size)
        self.require_cuda = bool(require_cuda)
        self.skip_self_neighbor_stages = tuple(
            sorted({int(stage) for stage in skip_self_neighbor_stages})
        )

    @torch.no_grad()
    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        points = data_dict.get("points")
        lengths = data_dict.get("lengths")



        if isinstance(points, (list, tuple)):
            return data_dict
        if not isinstance(points, torch.Tensor) or not isinstance(lengths, torch.Tensor):
            raise TypeError("Raw data_dict must contain tensor entries 'points' and 'lengths'")
        if self.require_cuda and not points.is_cuda:
            raise RuntimeError(
                "GPU preprocessing received CPU points. Move data_dict to CUDA before model(data_dict)."
            )
        if points.device != lengths.device:
            raise RuntimeError("points and lengths must be on the same device")

        hierarchy = precompute_stack_mode_gpu(
            points=points,
            lengths=lengths,
            num_stages=self.num_stages,
            voxel_size=self.voxel_size,
            num_neighbors=self.num_neighbors,
            subsample_ratio=self.subsample_ratio,
            query_chunk_size=self.query_chunk_size,
            support_chunk_size=self.support_chunk_size,
            skip_self_neighbor_stages=self.skip_self_neighbor_stages,
        )
        data_dict.update(hierarchy)
        return data_dict
