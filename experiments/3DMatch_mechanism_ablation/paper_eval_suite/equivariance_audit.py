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
from pareconv.datasets.registration.threedmatch.dataset import (
    ThreeDMatchPairDataset,
)
from pareconv.modules.ops import GPUStackModePreprocessor
from pareconv.utils.data import (
    build_dataloader_stack_mode,
    registration_collate_fn_stack_mode,
)
from pareconv.utils.torch import to_cuda


VECTOR_STAGE_ORDER = (
    "S2",
    "S3",
    "S4_pre_SPSA",
    "S4_post_SPSA",
    "D3",
    "D2_pre_MSPKI",
    "D2_post_MSPKI",
    "final_RE",
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure direct SO(3) equivariance on a frozen point hierarchy."
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--metadata_root", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], default="3DMatch")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "spsa", "mspki", "pki", "full"],
        default=["baseline", "full"],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--re_feature_source", choices=["official", "decoder", "encoder"], default="official")
    parser.add_argument("--rotation_seed", type=int, default=7351)
    parser.add_argument("--num_pairs", type=int, default=12)
    parser.add_argument("--num_rotations", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--query_chunk_size", type=int, default=1024)
    parser.add_argument("--support_chunk_size", type=int, default=4096)
    return parser


def checkpoint_path(args, variant: str) -> Path:
    return (
        Path(args.checkpoint_root)
        / variant
        / f"re_{args.re_feature_source}"
        / f"seed_{args.seed}"
        / "snapshots"
        / "best.pth.tar"
    )


def make_backbone(cfg, checkpoint: Path, dtype: torch.dtype) -> PAREConvFPN:
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in payload["model"].items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(backbone_state, strict=True)
    return backbone.to(device="cuda", dtype=dtype).eval()


def make_preprocessor(cfg) -> GPUStackModePreprocessor:
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


def make_loader(cfg, benchmark: str, num_pairs: int, num_workers: int):
    dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
        rotated=False,
    )
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive.")
    count = min(int(num_pairs), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64)
    subset = Subset(dataset, sorted(set(int(index) for index in indices)))
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


def quaternion_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def cast_hierarchy(data_dict: Dict, dtype: torch.dtype) -> Dict:
    casted = dict(data_dict)
    casted["points"] = [points.to(dtype=dtype) for points in data_dict["points"]]
    return casted


def rotate_hierarchy(data_dict: Dict, rotation: torch.Tensor) -> Dict:
    rotated = dict(data_dict)
    rotated["points"] = [points @ rotation.transpose(0, 1) for points in data_dict["points"]]
    return rotated


class StageCapture:
    def __init__(self, backbone: PAREConvFPN) -> None:
        self.values: Dict[str, torch.Tensor] = {}
        modules = {
            "S2": backbone.encoder2_3,
            "S3": backbone.encoder3_3,
            "S4_pre_SPSA": backbone.encoder4_3,
            "D3": backbone.decoder3,
            "D2_pre_MSPKI": backbone.decoder2,
            "final_RE": backbone.RE_head,
        }
        if backbone.spsa_scale is not None:
            modules["SPSA_update"] = backbone.spsa_scale
        if backbone.pki_scale is not None:
            modules["MSPKI_update"] = backbone.pki_scale
        self.handles = [module.register_forward_hook(self._hook(name)) for name, module in modules.items()]

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            self.values[name] = output.detach()

        return capture

    def run(self, backbone: PAREConvFPN, data_dict: Dict) -> Dict[str, torch.Tensor]:
        self.values.clear()
        outputs = backbone(data_dict)
        stages = dict(self.values)
        stages["S4_post_SPSA"] = stages["S4_pre_SPSA"] + stages.get(
            "SPSA_update", torch.zeros_like(stages["S4_pre_SPSA"])
        )
        stages["D2_post_MSPKI"] = stages["D2_pre_MSPKI"] + stages.get(
            "MSPKI_update", torch.zeros_like(stages["D2_pre_MSPKI"])
        )
        stages["fine_RI"] = outputs[1].detach()
        stages["coarse_RI"] = outputs[3].detach()
        return stages

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def relative_equivariance(
    original: torch.Tensor, rotated: torch.Tensor, rotation: torch.Tensor
) -> tuple[float, np.ndarray]:
    expected = original @ rotation.transpose(0, 1)
    difference = rotated - expected
    denominator = torch.linalg.vector_norm(original).clamp_min(torch.finfo(original.dtype).eps)
    global_error = torch.linalg.vector_norm(difference) / denominator
    point_denominator = torch.linalg.vector_norm(original, dim=(1, 2)).clamp_min(
        torch.finfo(original.dtype).eps
    )
    point_error = torch.linalg.vector_norm(difference, dim=(1, 2)) / point_denominator
    return float(global_error.item()), point_error.detach().cpu().numpy()


def relative_invariance(original: torch.Tensor, rotated: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(original).clamp_min(torch.finfo(original.dtype).eps)
    return float((torch.linalg.vector_norm(rotated - original) / denominator).item())


def pair_identity(data_dict: Dict, pair_index: int) -> Dict[str, str | int]:
    def scalar(value):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        return value

    return {
        "pair_index": int(pair_index),
        "scene": str(scalar(data_dict.get("scene_name", "unknown"))),
        "ref_frame": int(scalar(data_dict.get("ref_frame", -1))),
        "src_frame": int(scalar(data_dict.get("src_frame", -1))),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: Dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["variant"], row["dtype"], row["stage"])
        groups.setdefault(key, []).append(float(row["global_relative_error"]))
    summaries = []
    for (variant, dtype_name, stage), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        summaries.append(
            {
                "variant": variant,
                "dtype": dtype_name,
                "stage": stage,
                "observations": len(array),
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "p95": float(np.quantile(array, 0.95)),
                "maximum": float(array.max()),
            }
        )
    return summaries


def main() -> None:
    args = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen-hierarchy preprocessor.")
    if args.num_rotations < 1:
        raise ValueError("num_rotations must be positive.")

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
    rows: list[dict] = []

    for variant in args.variants:
        cfg.variant = variant
        _apply_variant(cfg)
        checkpoint = checkpoint_path(args, variant)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        for dtype_name in args.dtypes:
            dtype = dtype_map[dtype_name]
            backbone = make_backbone(cfg, checkpoint, dtype)
            capture = StageCapture(backbone)
            with torch.inference_mode():
                for pair_index, raw_data in enumerate(loader):
                    raw_data = to_cuda(raw_data)
                    hierarchy = preprocessor(raw_data)
                    hierarchy = cast_hierarchy(hierarchy, dtype)
                    identity = pair_identity(raw_data, pair_index)
                    original = capture.run(backbone, hierarchy)

                    for rotation_index, rotation_array in enumerate(rotations):
                        rotation = torch.as_tensor(rotation_array, device="cuda", dtype=dtype)
                        rotated = capture.run(backbone, rotate_hierarchy(hierarchy, rotation))
                        for stage in VECTOR_STAGE_ORDER:
                            global_error, point_errors = relative_equivariance(
                                original[stage], rotated[stage], rotation
                            )
                            rows.append(
                                {
                                    **identity,
                                    "benchmark": args.benchmark,
                                    "variant": variant,
                                    "dtype": dtype_name,
                                    "rotation_index": rotation_index,
                                    "stage": stage,
                                    "global_relative_error": global_error,
                                    "point_median_relative_error": float(np.median(point_errors)),
                                    "point_p95_relative_error": float(np.quantile(point_errors, 0.95)),
                                    "point_max_relative_error": float(point_errors.max()),
                                }
                            )
                        for stage in ("fine_RI", "coarse_RI"):
                            rows.append(
                                {
                                    **identity,
                                    "benchmark": args.benchmark,
                                    "variant": variant,
                                    "dtype": dtype_name,
                                    "rotation_index": rotation_index,
                                    "stage": stage,
                                    "global_relative_error": relative_invariance(
                                        original[stage], rotated[stage]
                                    ),
                                    "point_median_relative_error": math.nan,
                                    "point_p95_relative_error": math.nan,
                                    "point_max_relative_error": math.nan,
                                }
                            )
                    del hierarchy, original
                    torch.cuda.empty_cache()
            capture.close()
            del backbone
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "equivariance_observations.csv", rows)
    summary_rows = summarize(rows)
    write_csv(output_dir / "equivariance_summary.csv", summary_rows)
    with (output_dir / "equivariance_protocol.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "benchmark": args.benchmark,
                "variants": args.variants,
                "dtypes": args.dtypes,
                "seed": args.seed,
                "rotation_seed": args.rotation_seed,
                "requested_pairs": args.num_pairs,
                "evaluated_pair_batches": len({row["pair_index"] for row in rows}),
                "num_rotations": args.num_rotations,
                "hierarchy_policy": (
                    "The voxel hierarchy, neighbor indices, subsampling indices, and "
                    "upsampling indices are computed once and held fixed. The same SO(3) "
                    "rotation is then applied to every coordinate level."
                ),
                "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
            },
            stream,
            indent=2,
        )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
