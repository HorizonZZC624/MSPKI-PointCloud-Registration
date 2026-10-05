from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

SUITE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SUITE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (str(EXPERIMENT_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pareconv.engine import SingleTester
from pareconv.engine.base_tester import inject_default_parser
from pareconv.utils.common import ensure_dir, get_log_string
from pareconv.utils.torch import release_cuda

from config import get_benchmark_tag, make_cfg, make_experiment_parser
from loss import Evaluator
from model import create_model
from robust_dataset import build_paper_test_loader


def _sha256(file_name: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(file_name, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(payload: Dict, file_name: str) -> None:
    temporary = f"{file_name}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, file_name)


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description=(
            "Extract PARE-Net correspondences for official, rotated, density, "
            "and noise robustness evaluation."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="3DMatch",
        choices=["3DMatch", "3DLoMatch", "val"],
    )
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--perturbation_seed", type=int, default=7351)
    parser.add_argument("--min_points", type=int, default=64)
    parser.add_argument(
        "--condition_name",
        type=str,
        default="",
        help="Human-readable condition label stored in the manifest.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for paper statistics.",
    )
    parser.add_argument(
        "--overwrite_features",
        action="store_true",
        help="Delete existing extraction and registration outputs first.",
    )
    parser.add_argument(
        "--timing_warmup_pairs",
        type=int,
        default=5,
        help="Number of initial pairs excluded from mean forward-time statistics.",
    )
    parser.add_argument(
        "--save_full_features",
        action="store_true",
        help=(
            "Also save raw/fine/coarse point arrays and coarse descriptors. "
            "Disabled by default to reduce disk usage."
        ),
    )
    return inject_default_parser(parser)


class PaperTester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)

        start_time = time.time()
        data_loader, neighbor_limits = build_paper_test_loader(
            cfg,
            self.args.benchmark,
            keep_ratio=self.args.keep_ratio,
            noise_std=self.args.noise_std,
            perturbation_seed=self.args.perturbation_seed,
            min_points=self.args.min_points,
            max_pairs=self.args.max_pairs,
        )
        self.logger.info(f"Data loader created in {time.time() - start_time:.3f}s.")
        self.logger.info(f"KNN limits: {neighbor_limits}.")
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.total_params = sum(parameter.numel() for parameter in model.parameters())
        self.trainable_params = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        self.logger.critical(
            f"Variant={cfg.variant}, parameters={self.total_params:,}, "
            f"RE-source={cfg.fine_matching.re_feature_source}, "
            f"rotation={cfg.test.rotation_mode}, keep_ratio={self.args.keep_ratio}, "
            f"noise_std={self.args.noise_std}"
        )
        self.register_model(model)
        self.evaluator = Evaluator(cfg).cuda()

        self.benchmark_tag = get_benchmark_tag(cfg, self.args.benchmark)
        self.output_dir = osp.join(cfg.feature_dir, self.benchmark_tag)
        self.registration_output_dir = osp.join(
            cfg.registration_dir, self.benchmark_tag
        )
        if osp.isdir(self.output_dir):
            existing_files = list(Path(self.output_dir).rglob("*.npz"))
            manifest_exists = osp.isfile(osp.join(self.output_dir, "manifest.json"))
            if (existing_files or manifest_exists) and not self.args.overwrite_features:
                raise FileExistsError(
                    f"{self.output_dir} already contains extracted output. "
                    "Use --overwrite_features or let the orchestrator skip it."
                )
            if self.args.overwrite_features:
                shutil.rmtree(self.output_dir)

        stale_registration_files = (
            [path for path in Path(self.registration_output_dir).rglob("*") if path.is_file()]
            if osp.isdir(self.registration_output_dir)
            else []
        )
        if stale_registration_files and not self.args.overwrite_features:
            raise FileExistsError(
                f"{self.registration_output_dir} contains stale evaluation output. "
                "Use --overwrite_features."
            )
        if self.args.overwrite_features and osp.isdir(self.registration_output_dir):
            shutil.rmtree(self.registration_output_dir)

        ensure_dir(self.output_dir)
        self.manifest_file = osp.join(self.output_dir, "manifest.json")
        self.saved_pairs = 0
        self.expected_pairs = len(self.test_loader.dataset)
        self.manifest = None
        self.forward_times_ms = []

    def before_test_epoch(self):
        checkpoint_seed = int(self.checkpoint_metadata.get("seed", self.cfg.seed))
        self.manifest = {
            "code_version": getattr(self.cfg, "code_version", None),
            "runtime_code_version": getattr(self.cfg, "code_version", None),
            "checkpoint_code_version": self.checkpoint_metadata.get("code_version"),
            "variant": self.cfg.variant,
            "canonical_variant": self.cfg.canonical_variant,
            "seed": checkpoint_seed,
            "re_feature_source": self.cfg.fine_matching.re_feature_source,
            "coarse_context": self.cfg.backbone.coarse_context,
            "fine_context": self.cfg.backbone.fine_context,
            "spsa_ratio": float(self.cfg.backbone.spsa_partial_ratio),
            "spsa_qk_channels": int(self.cfg.backbone.spsa_qk_channels),
            "mspki_scales": list(self.cfg.backbone.mspki_neighbor_scales),
            "mspki_branch_channels": int(self.cfg.backbone.mspki_branch_channels),
            "layer_scale_init": float(self.cfg.backbone.layer_scale_init),
            "num_hypotheses": int(self.cfg.fine_matching.num_hypotheses),
            "benchmark": self.args.benchmark,
            "benchmark_tag": self.benchmark_tag,
            "rotation_mode": self.cfg.test.rotation_mode,
            "rotation_seed": self.cfg.test.rotation_seed,
            "condition_name": self.args.condition_name,
            "keep_ratio": float(self.args.keep_ratio),
            "noise_std": float(self.args.noise_std),
            "perturbation_seed": int(self.args.perturbation_seed),
            "max_pairs": self.args.max_pairs,
            "partial_dataset": self.args.max_pairs is not None,
            "snapshot": osp.abspath(self.args.snapshot),
            "snapshot_sha256": _sha256(self.args.snapshot),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_pairs": int(self.expected_pairs),
            "saved_pairs": 0,
            "completed": False,
            "parameters": int(self.total_params),
            "trainable_parameters": int(self.trainable_params),
            "timing_warmup_pairs": int(self.args.timing_warmup_pairs),
            "save_full_features": bool(self.args.save_full_features),
        }
        _atomic_json_dump(self.manifest, self.manifest_file)

    def test_step(self, iteration, data_dict):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output_dict = self.model(data_dict, compute_gt=True, estimate_transform=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output_dict["paper_eval_forward_time_ms"] = output_dict["estimated_transform"].new_tensor(
            elapsed_ms
        )
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        return self.evaluator(output_dict, data_dict)

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        del iteration
        message = (
            f"{data_dict['scene_name']}, id0={data_dict['ref_frame']}, "
            f"id1={data_dict['src_frame']}, "
            f"forward_ms={float(output_dict['paper_eval_forward_time_ms']):.2f}"
        )
        return message + ", " + get_log_string(result_dict=result_dict)

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        del result_dict
        scene_name = data_dict["scene_name"]
        ref_id = data_dict["ref_frame"]
        src_id = data_dict["src_frame"]
        forward_time_ms = float(release_cuda(output_dict["paper_eval_forward_time_ms"]))
        self.forward_times_ms.append(forward_time_ms)

        scene_dir = osp.join(self.output_dir, scene_name)
        ensure_dir(scene_dir)
        file_name = osp.join(scene_dir, f"{ref_id}_{src_id}.npz")
        payload = {
            "ref_node_corr_indices": release_cuda(output_dict["ref_node_corr_indices"]),
            "src_node_corr_indices": release_cuda(output_dict["src_node_corr_indices"]),
            "ref_corr_points": release_cuda(output_dict["ref_corr_points"]),
            "src_corr_points": release_cuda(output_dict["src_corr_points"]),
            "corr_scores": release_cuda(output_dict["corr_scores"]),
            "gt_node_corr_indices": release_cuda(output_dict["gt_node_corr_indices"]),
            "gt_node_corr_overlaps": release_cuda(output_dict["gt_node_corr_overlaps"]),
            "estimated_transform": release_cuda(output_dict["estimated_transform"]),
            "registration_attempted": release_cuda(output_dict["registration_attempted"]),
            "registration_succeeded": release_cuda(output_dict["registration_succeeded"]),
            "transform": release_cuda(data_dict["transform"]),
            "overlap": release_cuda(data_dict["overlap"]),
            "forward_time_ms": np.float64(forward_time_ms),
            "pair_iteration": np.int64(iteration),
            "keep_ratio": np.float32(self.args.keep_ratio),
            "noise_std": np.float32(self.args.noise_std),
        }
        if self.args.save_full_features:
            payload.update(
                {
                    "ref_points": release_cuda(output_dict["ref_points"]),
                    "src_points": release_cuda(output_dict["src_points"]),
                    "ref_points_f": release_cuda(output_dict["ref_points_f"]),
                    "src_points_f": release_cuda(output_dict["src_points_f"]),
                    "ref_points_c": release_cuda(output_dict["ref_points_c"]),
                    "src_points_c": release_cuda(output_dict["src_points_c"]),
                    "ref_feats_c": release_cuda(output_dict["ref_feats_c"]),
                    "src_feats_c": release_cuda(output_dict["src_feats_c"]),
                }
            )
        np.savez_compressed(file_name, **payload)
        self.saved_pairs += 1

    def after_test_epoch(self):
        warmup = max(0, int(self.args.timing_warmup_pairs))
        measured = self.forward_times_ms[warmup:] if len(self.forward_times_ms) > warmup else []
        self.manifest["saved_pairs"] = int(self.saved_pairs)
        self.manifest["completed"] = bool(self.saved_pairs == self.expected_pairs)
        self.manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manifest["mean_forward_time_ms"] = (
            float(np.mean(measured)) if measured else None
        )
        self.manifest["median_forward_time_ms"] = (
            float(np.median(measured)) if measured else None
        )
        self.manifest["std_forward_time_ms"] = (
            float(np.std(measured)) if measured else None
        )
        self.manifest["timed_pairs"] = int(len(measured))
        if torch.cuda.is_available():
            self.manifest["peak_cuda_memory_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024.0**2)
            )
        else:
            self.manifest["peak_cuda_memory_mb"] = None
        _atomic_json_dump(self.manifest, self.manifest_file)
        if not self.manifest["completed"]:
            raise RuntimeError(
                f"Feature extraction incomplete: saved={self.saved_pairs}, "
                f"expected={self.expected_pairs}."
            )


def main():
    parser = make_parser()
    known_args, _ = parser.parse_known_args()
    cfg = make_cfg(known_args)
    PaperTester(cfg, parser).run()


if __name__ == "__main__":
    main()
