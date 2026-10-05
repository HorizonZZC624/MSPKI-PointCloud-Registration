
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Subset

SUITE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SUITE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (str(EXPERIMENT_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backbone import PAREConvFPN
from config import make_cfg, _apply_variant
from pareconv.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from pareconv.modules.ops import GPUStackModePreprocessor
from pareconv.utils.data import build_dataloader_stack_mode, registration_collate_fn_stack_mode
from pareconv.utils.torch import to_cuda


VECTOR_STAGE_ORDER = (
    "S2",
    "S3",
    "S4_pre_context",
    "S4_post_context",
    "D3",
    "D2_pre_context",
    "D2_post_context",
    "final_RE",
)


def make_parser():
    p = argparse.ArgumentParser(description="Direct frozen-hierarchy SO(3) equivariance audit.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--metadata_root", required=True)
    p.add_argument("--checkpoint_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], default="3DLoMatch")
    p.add_argument("--variants", nargs="+", choices=["baseline", "spsa", "mspki", "pki", "full"],
                   default=["baseline", "mspki"])
    p.add_argument("--dtypes", nargs="+", choices=["float32", "float64"], default=["float32", "float64"])
    p.add_argument("--seed", type=int, default=7351)
    p.add_argument("--re_feature_source", choices=["official", "decoder", "encoder"], default="official")
    p.add_argument("--rotation_seed", type=int, default=7351)
    p.add_argument("--num_pairs", type=int, default=24)
    p.add_argument("--num_rotations", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--query_chunk_size", type=int, default=1024)
    p.add_argument("--support_chunk_size", type=int, default=4096)
    return p


def checkpoint_path(args, variant):
    return (Path(args.checkpoint_root) / variant / f"re_{args.re_feature_source}" /
            f"seed_{args.seed}" / "snapshots" / "best.pth.tar")


def make_backbone(cfg, checkpoint, dtype):
    backbone = PAREConvFPN(
        cfg.backbone.init_dim,
        cfg.backbone.output_dim,
        cfg.backbone.kernel_size,
        cfg.backbone.share_nonlinearity,
        cfg.backbone.conv_way,
        cfg.backbone.use_xyz,
        re_feature_source=cfg.fine_matching.re_feature_source,
        coarse_context=cfg.backbone.coarse_context,
        fine_context=cfg.backbone.fine_context,
        spsa_partial_ratio=cfg.backbone.spsa_partial_ratio,
        spsa_qk_channels=cfg.backbone.spsa_qk_channels,
        spsa_ffn_ratio=cfg.backbone.spsa_ffn_ratio,
        mspki_neighbor_scales=cfg.backbone.mspki_neighbor_scales,
        mspki_branch_channels=cfg.backbone.mspki_branch_channels,
        mspki_ffn_ratio=cfg.backbone.mspki_ffn_ratio,
        layer_scale_init=cfg.backbone.layer_scale_init,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    state = {
        k[len("backbone."):]: v
        for k, v in payload["model"].items()
        if k.startswith("backbone.")
    }
    backbone.load_state_dict(state, strict=True)
    return backbone.to(device="cuda", dtype=dtype).eval()


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


def quaternion_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def cast_hierarchy(data_dict: Dict, dtype):
    out = dict(data_dict)
    out["points"] = [x.to(dtype=dtype) for x in data_dict["points"]]
    return out


def rotate_hierarchy(data_dict: Dict, R):
    out = dict(data_dict)
    out["points"] = [x @ R.transpose(0, 1) for x in data_dict["points"]]
    return out


class StageCapture:
    def __init__(self, backbone):
        self.values = {}
        modules = {
            "S2": backbone.encoder2_3,
            "S3": backbone.encoder3_3,
            "S4_pre_context": backbone.encoder4_3,
            "D3": backbone.decoder3,
            "D2_pre_context": backbone.decoder2,
            "final_RE": backbone.RE_head,
        }
        if getattr(backbone, "coarse_context_scale", None) is not None:
            modules["coarse_context_update"] = backbone.coarse_context_scale
        if getattr(backbone, "fine_context_scale", None) is not None:
            modules["fine_context_update"] = backbone.fine_context_scale
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in modules.items()]

    def _hook(self, name):
        def fn(_module, _inputs, output):
            self.values[name] = output.detach()
        return fn

    def run(self, backbone, data_dict):
        self.values.clear()
        outputs = backbone(data_dict)
        stages = dict(self.values)
        stages["S4_post_context"] = stages["S4_pre_context"] + stages.get(
            "coarse_context_update", torch.zeros_like(stages["S4_pre_context"])
        )
        stages["D2_post_context"] = stages["D2_pre_context"] + stages.get(
            "fine_context_update", torch.zeros_like(stages["D2_pre_context"])
        )
        stages["fine_RI"] = outputs[1].detach()
        stages["coarse_RI"] = outputs[3].detach()
        return stages

    def close(self):
        for h in self.handles:
            h.remove()


def relative_equivariance(original, rotated, R):
    expected = original @ R.transpose(0, 1)
    diff = rotated - expected
    eps = torch.finfo(original.dtype).eps
    denom = torch.linalg.vector_norm(original).clamp_min(eps)
    global_error = torch.linalg.vector_norm(diff) / denom
    point_denom = torch.linalg.vector_norm(original, dim=(1, 2)).clamp_min(eps)
    point_error = torch.linalg.vector_norm(diff, dim=(1, 2)) / point_denom
    return float(global_error.item()), point_error.detach().cpu().numpy()


def relative_invariance(original, rotated):
    eps = torch.finfo(original.dtype).eps
    denom = torch.linalg.vector_norm(original).clamp_min(eps)
    return float((torch.linalg.vector_norm(rotated - original) / denom).item())


def scalar(v):
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return v[0]
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    return v


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    cfg = make_cfg()
    cfg.data.dataset_root = str(Path(args.dataset_root).resolve())
    cfg.data.metadata_root = str(Path(args.metadata_root).resolve())
    cfg.test.num_workers = int(args.num_workers)
    cfg.preprocess.query_chunk_size = int(args.query_chunk_size)
    cfg.preprocess.support_chunk_size = int(args.support_chunk_size)
    cfg.fine_matching.re_feature_source = args.re_feature_source

    loader = make_loader(cfg, args.benchmark, args.num_pairs, args.num_workers)
    preprocessor = make_preprocessor(cfg)
    rng = np.random.default_rng(args.rotation_seed)
    rotations = [quaternion_rotation(rng) for _ in range(args.num_rotations)]
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    rows = []

    for variant in args.variants:
        cfg.variant = variant
        _apply_variant(cfg)
        ckpt = checkpoint_path(args, variant)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)

        for dtype_name in args.dtypes:
            dtype = dtype_map[dtype_name]
            backbone = make_backbone(cfg, ckpt, dtype)
            capture = StageCapture(backbone)

            with torch.inference_mode():
                for pair_index, raw in enumerate(loader):
                    raw = to_cuda(raw)
                    hierarchy = cast_hierarchy(preprocessor(raw), dtype)
                    original = capture.run(backbone, hierarchy)
                    identity = {
                        "pair_index": pair_index,
                        "scene": str(scalar(raw.get("scene_name", "unknown"))),
                        "ref_frame": int(scalar(raw.get("ref_frame", -1))),
                        "src_frame": int(scalar(raw.get("src_frame", -1))),
                    }

                    for ridx, R_np in enumerate(rotations):
                        R = torch.as_tensor(R_np, device="cuda", dtype=dtype)
                        rotated = capture.run(backbone, rotate_hierarchy(hierarchy, R))

                        for stage in VECTOR_STAGE_ORDER:
                            ge, pe = relative_equivariance(original[stage], rotated[stage], R)
                            rows.append({
                                **identity,
                                "benchmark": args.benchmark,
                                "variant": variant,
                                "dtype": dtype_name,
                                "rotation_index": ridx,
                                "stage": stage,
                                "global_relative_error": ge,
                                "point_median_relative_error": float(np.median(pe)),
                                "point_p95_relative_error": float(np.quantile(pe, 0.95)),
                                "point_max_relative_error": float(pe.max()),
                            })

                        for stage in ("fine_RI", "coarse_RI"):
                            rows.append({
                                **identity,
                                "benchmark": args.benchmark,
                                "variant": variant,
                                "dtype": dtype_name,
                                "rotation_index": ridx,
                                "stage": stage,
                                "global_relative_error": relative_invariance(original[stage], rotated[stage]),
                                "point_median_relative_error": math.nan,
                                "point_p95_relative_error": math.nan,
                                "point_max_relative_error": math.nan,
                            })

            capture.close()
            del backbone
            torch.cuda.empty_cache()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "equivariance_observations.csv", rows)

    summary = []
    for variant in sorted({r["variant"] for r in rows}):
        for dtype in sorted({r["dtype"] for r in rows if r["variant"] == variant}):
            for stage in sorted({r["stage"] for r in rows if r["variant"] == variant and r["dtype"] == dtype}):
                vals = np.asarray([
                    float(r["global_relative_error"]) for r in rows
                    if r["variant"] == variant and r["dtype"] == dtype and r["stage"] == stage
                ], dtype=np.float64)
                summary.append({
                    "variant": variant,
                    "dtype": dtype,
                    "stage": stage,
                    "observations": len(vals),
                    "mean": float(vals.mean()),
                    "median": float(np.median(vals)),
                    "p95": float(np.quantile(vals, 0.95)),
                    "maximum": float(vals.max()),
                })

    write_csv(out / "equivariance_summary.csv", summary)
    with (out / "equivariance_protocol.json").open("w", encoding="utf-8") as f:
        json.dump({
            "benchmark": args.benchmark,
            "variants": args.variants,
            "dtypes": args.dtypes,
            "model_seed": args.seed,
            "rotation_seed": args.rotation_seed,
            "num_pairs": args.num_pairs,
            "num_rotations": args.num_rotations,
            "hierarchy_policy": "Hierarchy and neighbor indices are computed once and frozen; the same SO(3) rotation is applied to all coordinate levels.",
            "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        }, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
