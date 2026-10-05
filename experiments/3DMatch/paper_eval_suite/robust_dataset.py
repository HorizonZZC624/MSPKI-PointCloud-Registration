from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset, Subset

from pareconv.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from pareconv.utils.data import (
    build_dataloader_stack_mode,
    registration_collate_fn_stack_mode,
)


def _stable_seed(base_seed: int, index: int, key: str) -> int:
    payload = f"{int(base_seed)}:{int(index)}:{key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


class DeterministicPerturbationDataset(Dataset):
    pass






    def __init__(
        self,
        base_dataset: Dataset,
        keep_ratio: float = 1.0,
        noise_std: float = 0.0,
        perturbation_seed: int = 7351,
        min_points: int = 64,
    ) -> None:
        if not 0.0 < float(keep_ratio) <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if float(noise_std) < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if int(min_points) < 3:
            raise ValueError("min_points must be at least 3.")

        self.base_dataset = base_dataset
        self.keep_ratio = float(keep_ratio)
        self.noise_std = float(noise_std)
        self.perturbation_seed = int(perturbation_seed)
        self.min_points = int(min_points)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _perturb_points(self, points: Any, index: int, key: str) -> np.ndarray:
        array = np.asarray(points, dtype=np.float32).copy()
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(
                f"{key} must have shape (N, >=3), got {tuple(array.shape)}."
            )
        if array.shape[0] == 0:
            raise ValueError(f"{key} is empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or infinity.")

        rng = np.random.default_rng(_stable_seed(self.perturbation_seed, index, key))

        if self.keep_ratio < 1.0:
            keep_count = int(round(array.shape[0] * self.keep_ratio))
            keep_count = max(min(self.min_points, array.shape[0]), keep_count)
            keep_count = min(keep_count, array.shape[0])
            if keep_count < array.shape[0]:
                selected = np.sort(
                    rng.choice(array.shape[0], size=keep_count, replace=False)
                )
                array = array[selected]

        if self.noise_std > 0.0:
            noise = rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=(array.shape[0], 3),
            ).astype(np.float32)
            array[:, :3] += noise

        return array

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, dict):
            raise TypeError(
                "ThreeDMatchPairDataset is expected to return a dictionary."
            )
        sample = copy.deepcopy(sample)
        for key in ("ref_points", "src_points"):
            if key not in sample:
                raise KeyError(f"Dataset sample does not contain {key!r}.")
            sample[key] = self._perturb_points(sample[key], index, key)
        return sample


def build_paper_test_loader(
    cfg,
    benchmark: str,
    keep_ratio: float = 1.0,
    noise_std: float = 0.0,
    perturbation_seed: int = 7351,
    min_points: int = 64,
    max_pairs: Optional[int] = None,
    pair_indices: Optional[list[int]] = None,
):
    base_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
        rotated=cfg.test.rotation_mode == "so3",
        rotation_seed=cfg.test.rotation_seed,
    )

    dataset: Dataset = DeterministicPerturbationDataset(
        base_dataset,
        keep_ratio=keep_ratio,
        noise_std=noise_std,
        perturbation_seed=perturbation_seed,
        min_points=min_points,
    )
    if pair_indices is not None:
        indices = [int(i) for i in pair_indices]
        if any(i < 0 or i >= len(dataset) for i in indices):
            raise IndexError("pair_indices contains an out-of-range index.")
        dataset = Subset(dataset, indices)
    if max_pairs is not None:
        if int(max_pairs) < 1:
            raise ValueError("max_pairs must be positive when provided.")
        dataset = Subset(dataset, list(range(min(int(max_pairs), len(dataset)))))

    loader = build_dataloader_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        precompute_data=False,
    )
    return loader, cfg.backbone.num_neighbors
