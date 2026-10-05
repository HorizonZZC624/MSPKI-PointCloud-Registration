
import hashlib

import numpy as np
from torch.utils.data import Dataset

from pareconv.datasets.registration.kitti.dataset import OdometryKittiPairDataset
from pareconv.utils.data import registration_collate_fn_stack_mode, build_dataloader_stack_mode


def train_valid_data_loader(cfg, distributed):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        "train",
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        augmentation_crop=cfg.train.augmentation_crop,
        point_keep_ratio=cfg.train.point_keep_ratio,
        matching_radius=cfg.train.matching_radius,
    )
    valid_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        "val",
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
    )

    common = dict(
        collate_fn=registration_collate_fn_stack_mode,
        num_stages=cfg.backbone.num_stages,
        voxel_size=cfg.backbone.init_voxel_size,
        num_neighbors=cfg.backbone.num_neighbors,
        subsample_ratio=cfg.backbone.subsample_ratio,
        batch_size=1,
        precompute_data=False,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        common.pop("collate_fn"),
        common.pop("num_stages"),
        common.pop("voxel_size"),
        common.pop("num_neighbors"),
        common.pop("subsample_ratio"),
        batch_size=1,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=False,
    )
    return train_loader, valid_loader, cfg.backbone.num_neighbors


class FixedSourceDensityDataset(Dataset):
    pass

    def __init__(self, base_dataset, keep_ratio, mask_seed):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if mask_seed < 0:
            raise ValueError("mask_seed must be nonnegative.")
        self.base_dataset = base_dataset
        self.keep_ratio = float(keep_ratio)
        self.mask_seed = int(mask_seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        if self.keep_ratio == 1.0:
            return sample

        points = sample["src_points"]
        feats = sample["src_feats"]
        if len(points) != len(feats) or len(points) == 0:
            raise ValueError("Source points and features must have the same nonzero length.")

        scan_key = f"{self.mask_seed}:{sample['seq_id']}:{sample['src_frame']}"
        digest = hashlib.sha256(scan_key.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        keep_count = max(1, round(len(points) * self.keep_ratio))
        selected = np.sort(rng.permutation(len(points))[:keep_count])
        sample = dict(sample)
        sample["src_points"] = points[selected].copy()
        sample["src_feats"] = feats[selected].copy()
        return sample


def test_data_loader(cfg, *, density_keep_ratio=1.0, density_mask_seed=9017):
    test_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        "test",
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
    )
    test_dataset = FixedSourceDensityDataset(
        test_dataset, density_keep_ratio, density_mask_seed
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        precompute_data=False,
    )
    return test_loader, cfg.backbone.num_neighbors
