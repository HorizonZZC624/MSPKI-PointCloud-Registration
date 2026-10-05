from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Subset


SUITE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SUITE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (str(EXPERIMENT_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import make_cfg, _apply_variant
from model import create_model
from pareconv.datasets.registration.threedmatch.dataset import (
    ThreeDMatchPairDataset,
)
from pareconv.utils.data import (
    build_dataloader_stack_mode,
    registration_collate_fn_stack_mode,
)
from pareconv.utils.torch import to_cuda


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA-synchronized component and end-to-end inference benchmark."
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
        default=["baseline", "spsa", "mspki", "full"],
    )
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--re_feature_source", choices=["official", "decoder", "encoder"], default="official")
    parser.add_argument("--num_pairs", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
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
    count = min(max(1, int(num_pairs)), len(dataset))
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


def load_model(cfg, checkpoint: Path):
    model = create_model(cfg).cuda().eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model


class ComponentTimer:
    def __init__(self, module: torch.nn.Module) -> None:
        self.start_event = None
        self.end_event = None
        self.pre_handle = module.register_forward_pre_hook(self._start)
        self.post_handle = module.register_forward_hook(self._stop)

    def _start(self, _module, _inputs):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def _stop(self, _module, _inputs, _output):
        self.end_event.record()

    def elapsed_ms(self) -> float:
        if self.start_event is None or self.end_event is None:
            return math.nan
        return float(self.start_event.elapsed_time(self.end_event))

    def clear(self) -> None:
        self.start_event = None
        self.end_event = None

    def close(self) -> None:
        self.pre_handle.remove()
        self.post_handle.remove()


class CallableTimer:
    def __init__(self, callable_object) -> None:
        self.callable_object = callable_object
        self.start_event = None
        self.end_event = None

    def __call__(self, *args, **kwargs):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        output = self.callable_object(*args, **kwargs)
        self.end_event.record()
        return output

    def elapsed_ms(self) -> float:
        if self.start_event is None or self.end_event is None:
            return math.nan
        return float(self.start_event.elapsed_time(self.end_event))

    def clear(self) -> None:
        self.start_event = None
        self.end_event = None

    def close(self) -> None:
        return None


def identity(data_dict: Dict, pair_index: int) -> Dict[str, str | int]:
    def scalar(value):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        return value

    return {
        "pair_index": pair_index,
        "scene": str(scalar(data_dict.get("scene_name", "unknown"))),
        "ref_frame": int(scalar(data_dict.get("ref_frame", -1))),
        "src_frame": int(scalar(data_dict.get("src_frame", -1))),
    }


def fresh_pair(data_dict: Dict) -> Dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data_dict.items()
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        summary = {
            "variant": variant,
            "observations": len(selected),
            "parameters": selected[0]["parameters"],
            "checkpoint_MiB": selected[0]["checkpoint_MiB"],
            "peak_memory_MiB": max(float(row["peak_memory_MiB"]) for row in selected),
        }
        for column in ("preprocess_ms", "backbone_ms", "fhp_ms", "total_ms"):
            values = np.asarray(
                [float(row[column]) for row in selected if math.isfinite(float(row[column]))],
                dtype=np.float64,
            )
            summary[f"{column}_median"] = float(np.median(values))
            summary[f"{column}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{column}_q75"] = float(np.quantile(values, 0.75))
            summary[f"{column}_p95"] = float(np.quantile(values, 0.95))
        output.append(summary)
    return output


def main() -> None:
    args = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative.")

    cfg = make_cfg()
    cfg.data.dataset_root = str(Path(args.dataset_root).resolve())
    cfg.data.metadata_root = str(Path(args.metadata_root).resolve())
    cfg.test.num_workers = int(args.num_workers)
    cfg.preprocess.query_chunk_size = int(args.query_chunk_size)
    cfg.preprocess.support_chunk_size = int(args.support_chunk_size)
    cfg.fine_matching.re_feature_source = args.re_feature_source
    loader = make_loader(cfg, args.benchmark, args.num_pairs, args.num_workers)
    raw_pairs = [to_cuda(data_dict) for data_dict in loader]
    rows: list[dict] = []

    for variant in args.variants:
        cfg.variant = variant
        _apply_variant(cfg)
        checkpoint = checkpoint_path(args, variant)
        model = load_model(cfg, checkpoint)
        preprocess_timer = CallableTimer(model.preprocessor)
        model.preprocessor = preprocess_timer
        timers = {
            "preprocess_ms": preprocess_timer,
            "backbone_ms": ComponentTimer(model.backbone),
            "fhp_ms": ComponentTimer(model.fine_matching),
        }
        with torch.inference_mode():
            for warmup_index in range(args.warmup):
                model(
                    fresh_pair(raw_pairs[warmup_index % len(raw_pairs)]),
                    compute_gt=False,
                    estimate_transform=True,
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            for pair_index, raw_data in enumerate(raw_pairs):
                pair_id = identity(raw_data, pair_index)
                for repeat in range(args.repeats):
                    for timer in timers.values():
                        timer.clear()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = model(
                        fresh_pair(raw_data), compute_gt=False, estimate_transform=True
                    )
                    torch.cuda.synchronize()
                    total_ms = 1000.0 * (time.perf_counter() - start)
                    rows.append(
                        {
                            **pair_id,
                            "benchmark": args.benchmark,
                            "variant": variant,
                            "repeat": repeat,
                            "preprocess_ms": timers["preprocess_ms"].elapsed_ms(),
                            "backbone_ms": timers["backbone_ms"].elapsed_ms(),
                            "fhp_ms": timers["fhp_ms"].elapsed_ms(),
                            "total_ms": total_ms,
                            "registration_succeeded": int(
                                bool(output["registration_succeeded"].item())
                            ),
                            "parameters": sum(parameter.numel() for parameter in model.parameters()),
                            "checkpoint_MiB": checkpoint.stat().st_size / (1024**2),
                            "peak_memory_MiB": torch.cuda.max_memory_allocated() / (1024**2),
                        }
                    )
        for timer in timers.values():
            timer.close()
        del model
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "efficiency_observations.csv", rows)
    summary_rows = summarize(rows)
    write_csv(output_dir / "efficiency_summary.csv", summary_rows)
    with (output_dir / "efficiency_environment.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "gpu": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "benchmark": args.benchmark,
                "num_pairs": len(raw_pairs),
                "repeats": args.repeats,
                "warmup": args.warmup,
                "num_hypotheses": cfg.fine_matching.num_hypotheses,
                "timing": "CUDA events for components; synchronized wall time for total",
                "data_loading_included": False,
                "ground_truth_computation_included": False,
            },
            stream,
            indent=2,
        )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
