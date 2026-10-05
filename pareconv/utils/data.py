

from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
import torch

from pareconv.modules.ops import grid_subsample, radius_search
from pareconv.utils.torch import build_dataloader


def precompute_subsample(
    points: torch.Tensor,
    lengths: torch.Tensor,
    num_stages: int,
    voxel_size: float,
    num_neighbors: Sequence[int],
    subsample_ratio: float,
):
    if num_stages < 1 or num_stages != len(num_neighbors):
        raise ValueError('num_stages must be positive and equal len(num_neighbors).')
    if voxel_size <= 0 or subsample_ratio <= 1:
        raise ValueError('voxel_size must be positive and subsample_ratio > 1.')

    points_list = []
    lengths_list = []
    for stage in range(num_stages):
        if stage > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= subsample_ratio
    return {'points': points_list, 'lengths': lengths_list}


def precompute_neighbors(points_list, lengths_list, num_stages, num_neighbors):
    if num_stages != len(points_list) or num_stages != len(lengths_list):
        raise ValueError('Point hierarchy length does not match num_stages.')

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []
    for stage in range(num_stages):
        current_points = points_list[stage]
        current_lengths = lengths_list[stage]
        neighbors_list.append(
            radius_search(
                current_points,
                current_points,
                current_lengths,
                current_lengths,
                int(num_neighbors[stage]),
            )
        )

        if stage < num_stages - 1:
            subsampled_points = points_list[stage + 1]
            subsampled_lengths = lengths_list[stage + 1]
            subsampling_list.append(
                radius_search(
                    subsampled_points,
                    current_points,
                    subsampled_lengths,
                    current_lengths,
                    int(num_neighbors[stage]),
                )
            )

            if stage > 0:
                upsampling_list.append(
                    radius_search(
                        current_points,
                        subsampled_points,
                        current_lengths,
                        subsampled_lengths,
                        1,
                    )
                )

    return {
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }



precompute_neibors = precompute_neighbors


def registration_collate_fn_stack_mode(
    data_dicts,
    num_stages,
    voxel_size,
    num_neighbors,
    subsample_ratio,
    precompute_data=True,
):
    pass





    if not data_dicts:
        raise ValueError('Cannot collate an empty batch.')
    batch_size = len(data_dicts)
    collated_dict = {}
    for data_dict in data_dicts:
        required = {'ref_feats', 'src_feats', 'ref_points', 'src_points'}
        missing = required.difference(data_dict)
        if missing:
            raise KeyError(f'Missing required sample keys: {sorted(missing)}')
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            collated_dict.setdefault(key, []).append(value)

    ref_feats = collated_dict.pop('ref_feats')
    src_feats = collated_dict.pop('src_feats')
    ref_points = collated_dict.pop('ref_points')
    src_points = collated_dict.pop('src_points')
    feats = torch.cat(ref_feats + src_feats, dim=0)
    points_list = ref_points + src_points
    lengths = torch.tensor(
        [point_cloud.shape[0] for point_cloud in points_list], dtype=torch.long
    )
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        collated_dict = {key: value[0] for key, value in collated_dict.items()}

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_subsample(
            points,
            lengths,
            num_stages,
            voxel_size,
            num_neighbors,
            subsample_ratio,
        )
        input_dict.update(
            precompute_neighbors(
                input_dict['points'],
                input_dict['lengths'],
                num_stages,
                num_neighbors,
            )
        )
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size
    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    num_neighbors,
    subsample_ratio,
    keep_ratio=0.8,
    sample_threshold=2000,
):
    pass





    if not 0 < keep_ratio <= 1:
        raise ValueError('keep_ratio must be in (0, 1].')
    max_limit = max(int(value) for value in num_neighbors)
    histogram_size = max(max_limit + 1, 256)
    histograms = np.zeros((num_stages, histogram_size), dtype=np.int64)

    for index in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[index]],
            num_stages=num_stages,
            voxel_size=voxel_size,
            num_neighbors=num_neighbors,
            subsample_ratio=subsample_ratio,
            precompute_data=True,
        )
        for stage, neighbors in enumerate(data_dict['neighbors']):
            sentinel = data_dict['points'][stage].shape[0]
            counts = (neighbors.cpu().numpy() < sentinel).sum(axis=1)
            histograms[stage] += np.bincount(
                np.minimum(counts, histogram_size - 1),
                minlength=histogram_size,
            )
        if np.min(histograms.sum(axis=1)) >= sample_threshold:
            break

    limits = []
    for histogram in histograms:
        total = histogram.sum()
        if total == 0:
            limits.append(max_limit)
            continue
        cumulative = np.cumsum(histogram)
        limits.append(int(np.searchsorted(cumulative, keep_ratio * total) + 1))
    return np.asarray(limits, dtype=np.int64)


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    num_neighbors,
    subsample_ratio,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            num_neighbors=num_neighbors,
            subsample_ratio=subsample_ratio,
            precompute_data=precompute_data,
        ),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        distributed=distributed,
    )
