
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

SUITE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SUITE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for p in (str(EXPERIMENT_DIR), str(PROJECT_ROOT), str(SUITE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import make_cfg
from pareconv.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from pareconv.modules.ops import GPUStackModePreprocessor
from pareconv.utils.data import build_dataloader_stack_mode, registration_collate_fn_stack_mode
from pareconv.utils.torch import to_cuda


def quaternion_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def make_loader(cfg, benchmark, num_pairs, num_workers):
    ds = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
        rotated=False,
    )
    count = min(max(1, int(num_pairs)), len(ds))
    idx = np.linspace(0, len(ds) - 1, num=count, dtype=np.int64)
    subset = Subset(ds, sorted(set(int(i) for i in idx)))
    return build_dataloader_stack_mode(
        subset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=1,
        num_workers=int(num_workers),
        shuffle=False,
        precompute_data=False,
    )


def make_preprocessor(cfg):
    return GPUStackModePreprocessor(
        num_stages=cfg.backbone.num_stages,
        voxel_size=cfg.backbone.init_voxel_size,
        num_neighbors=cfg.backbone.num_neighbors,
        subsample_ratio=cfg.backbone.subsample_ratio,
        query_chunk_size=cfg.preprocess.query_chunk_size,
        support_chunk_size=cfg.preprocess.support_chunk_size,
        require_cuda=True,
        skip_self_neighbor_stages=(0,),
    )


def squared_distances(points, neighbors):
    rel = points[neighbors] - points[:, None]
    return (rel * rel).sum(dim=-1)


def prefix_set_equal(a, b, k):
    aa = torch.sort(a[:, :k], dim=1).values
    bb = torch.sort(b[:, :k], dim=1).values
    return (aa == bb).all(dim=1)


def main():
    p = argparse.ArgumentParser(description="Audit whether incoming KNN order is already usable and whether MSPKI re-sorting changes scale-boundary membership under rotation.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--metadata_root", required=True)
    p.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], default="3DLoMatch")
    p.add_argument("--num_pairs", type=int, default=24)
    p.add_argument("--num_rotations", type=int, default=4)
    p.add_argument("--rotation_seed", type=int, default=7351)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--stage", type=int, default=1, help="Fine-context hierarchy stage; default=1.")
    p.add_argument("--scales", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    cfg = make_cfg()
    cfg.data.dataset_root = str(Path(args.dataset_root).resolve())
    cfg.data.metadata_root = str(Path(args.metadata_root).resolve())
    cfg.test.num_workers = args.num_workers

    loader = make_loader(cfg, args.benchmark, args.num_pairs, args.num_workers)
    pre = make_preprocessor(cfg)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rng = np.random.default_rng(args.rotation_seed)
    rotations = [quaternion_rotation(rng) for _ in range(args.num_rotations)]

    totals = {
        "rows": 0,
        "incoming_exact_non_decreasing_rows": 0,
        "prefix_matches_precomputed_vs_resorted": {str(k): 0 for k in args.scales},
        "rotation_prefix_unchanged": {str(k): 0 for k in args.scales},
        "rotation_prefix_total": {str(k): 0 for k in args.scales},
        "boundary_gap_zero_or_tiny": {str(k): 0 for k in args.scales},
        "boundary_gap_total": {str(k): 0 for k in args.scales},
    }

    tiny = 1e-12 if dtype == torch.float64 else 1e-6

    with torch.inference_mode():
        for raw in loader:
            raw = to_cuda(raw)
            h = pre(raw)
            points = h["points"][args.stage].to(dtype=dtype)
            neighbors = h["neighbors"][args.stage]

            d0 = squared_distances(points, neighbors)
            order0 = torch.argsort(d0, dim=1)
            sorted_neighbors0 = torch.gather(neighbors, 1, order0)

            rows = neighbors.shape[0]
            totals["rows"] += int(rows)

            if neighbors.shape[1] > 1:
                monotonic = (d0[:, 1:] >= d0[:, :-1]).all(dim=1)
                totals["incoming_exact_non_decreasing_rows"] += int(monotonic.sum().item())

            for k in args.scales:
                if k <= neighbors.shape[1]:
                    eq = prefix_set_equal(neighbors, sorted_neighbors0, k)
                    totals["prefix_matches_precomputed_vs_resorted"][str(k)] += int(eq.sum().item())

                    if k < neighbors.shape[1]:
                        sd = torch.gather(d0, 1, order0)
                        gap = sd[:, k] - sd[:, k - 1]
                        scale = sd[:, k].abs().clamp_min(torch.finfo(dtype).eps)
                        rel_gap = gap.abs() / scale
                        totals["boundary_gap_zero_or_tiny"][str(k)] += int((rel_gap <= tiny).sum().item())
                        totals["boundary_gap_total"][str(k)] += int(rows)

            for R_np in rotations:
                R = torch.as_tensor(R_np, device=points.device, dtype=dtype)
                rp = points @ R.transpose(0, 1)
                dr = squared_distances(rp, neighbors)
                orderr = torch.argsort(dr, dim=1)
                sorted_neighborsr = torch.gather(neighbors, 1, orderr)

                for k in args.scales:
                    if k <= neighbors.shape[1]:
                        eq = prefix_set_equal(sorted_neighbors0, sorted_neighborsr, k)
                        totals["rotation_prefix_unchanged"][str(k)] += int(eq.sum().item())
                        totals["rotation_prefix_total"][str(k)] += int(rows)

    summary = {
        "benchmark": args.benchmark,
        "dtype": args.dtype,
        "stage": args.stage,
        "num_pairs": args.num_pairs,
        "num_rotations": args.num_rotations,
        "rotation_seed": args.rotation_seed,
        "scales": args.scales,
        "rows": totals["rows"],
        "incoming_exact_non_decreasing_fraction": (
            totals["incoming_exact_non_decreasing_rows"] / max(totals["rows"], 1)
        ),
        "precomputed_prefix_matches_resorted_fraction": {},
        "rotation_prefix_unchanged_fraction": {},
        "boundary_gap_zero_or_tiny_fraction": {},
    }

    for k in args.scales:
        key = str(k)
        summary["precomputed_prefix_matches_resorted_fraction"][key] = (
            totals["prefix_matches_precomputed_vs_resorted"][key] / max(totals["rows"], 1)
        )
        summary["rotation_prefix_unchanged_fraction"][key] = (
            totals["rotation_prefix_unchanged"][key] / max(totals["rotation_prefix_total"][key], 1)
        )
        summary["boundary_gap_zero_or_tiny_fraction"][key] = (
            totals["boundary_gap_zero_or_tiny"][key] / max(totals["boundary_gap_total"][key], 1)
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
