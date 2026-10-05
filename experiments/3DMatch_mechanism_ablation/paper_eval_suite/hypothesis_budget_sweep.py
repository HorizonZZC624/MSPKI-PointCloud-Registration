from __future__ import annotations

import csv
import json
import os
import pickle
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

from pareconv.datasets.registration.threedmatch.utils import (
    compute_transform_error,
    get_gt_logs_and_infos,
    get_num_fragments,
    get_scene_abbr,
)
from pareconv.engine import SingleTester
from pareconv.engine.base_tester import inject_default_parser
from pareconv.utils.torch import release_cuda, to_cuda

from config import make_cfg, make_experiment_parser
from model import create_model
from robust_dataset import build_paper_test_loader


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description=(
            "Evaluate several FHP budgets in one backbone pass per pair and "
            "write compact official covariance-RR summaries."
        ),
    )
    parser.add_argument(
        "--benchmark", choices=["3DMatch", "3DLoMatch"], required=True
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[100, 250, 500, 1000]
    )
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument(
        "--pairs_per_scene",
        type=int,
        default=None,
        help="Use a deterministic, evenly spaced scene-stratified diagnostic subset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return inject_default_parser(parser)


def _safe_remove_tree(path: Path, allowed_root: Path) -> None:
    target = path.resolve()
    root = allowed_root.resolve()
    if target == root or os.path.commonpath([str(target), str(root)]) != str(root):
        raise RuntimeError(f"Unsafe recursive-delete target: {target}")
    shutil.rmtree(target)


def _write_json(path: Path, payload: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


class HypothesisBudgetTester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        self.budgets = sorted(set(int(value) for value in self.args.budgets))
        if not self.budgets or self.budgets[0] < 1:
            raise ValueError("All hypothesis budgets must be positive.")
        if self.args.max_pairs is not None and self.args.pairs_per_scene is not None:
            raise ValueError("max_pairs and pairs_per_scene are mutually exclusive.")

        pair_indices = None
        if self.args.pairs_per_scene is not None:
            count = int(self.args.pairs_per_scene)
            if count < 1:
                raise ValueError("pairs_per_scene must be positive.")
            metadata_path = Path(cfg.data.metadata_root) / f"{self.args.benchmark}.pkl"
            with metadata_path.open("rb") as stream:
                metadata = pickle.load(stream)
            grouped: Dict[str, list[int]] = {}
            for index, row in enumerate(metadata):
                grouped.setdefault(row["scene_name"], []).append(index)
            pair_indices = []
            for scene_name in sorted(grouped):
                gt_root = (
                    Path(cfg.data.metadata_root)
                    / "benchmarks"
                    / self.args.benchmark
                    / scene_name
                )
                gt_indices, _, _ = get_gt_logs_and_infos(
                    str(gt_root), get_num_fragments(scene_name)
                )
                scene_indices = [
                    index
                    for index in grouped[scene_name]
                    if int(
                        gt_indices[
                            int(metadata[index]["frag_id0"]),
                            int(metadata[index]["frag_id1"]),
                        ]
                    )
                    != -1
                ]
                positions = np.linspace(
                    0, len(scene_indices) - 1, min(count, len(scene_indices)), dtype=int
                )
                pair_indices.extend(scene_indices[position] for position in positions)
            pair_indices = sorted(set(pair_indices))

        data_loader, neighbor_limits = build_paper_test_loader(
            cfg,
            self.args.benchmark,
            keep_ratio=1.0,
            noise_std=0.0,
            perturbation_seed=cfg.seed,
            min_points=64,
            max_pairs=self.args.max_pairs,
            pair_indices=pair_indices,
        )
        self.logger.info(f"KNN limits: {neighbor_limits}.")
        self.register_loader(data_loader)
        self.register_model(create_model(cfg).cuda())

        self.expected_pairs = len(self.test_loader.dataset)
        self.results_root = Path(self.args.results_root).resolve()
        self.output_roots: Dict[int, Path] = {}
        for budget in self.budgets:
            run_root = (
                self.results_root
                / f"fhp_{budget}"
                / f"standard_{self.args.benchmark}"
                / cfg.variant
                / f"re_{cfg.fine_matching.re_feature_source}"
                / f"seed_{cfg.seed}"
            )
            if self.args.overwrite and run_root.is_dir():
                _safe_remove_tree(run_root, self.results_root)
            elif run_root.exists():
                raise FileExistsError(f"Output already exists: {run_root}")
            output_root = run_root / "registration" / self.args.benchmark
            output_root.mkdir(parents=True, exist_ok=True)
            self.output_roots[budget] = output_root

        self.rows = {budget: [] for budget in self.budgets}
        self.backbone_times_ms = []
        self.fhp_times_ms = {budget: [] for budget in self.budgets}
        self.scene_ground_truth = {}

    def run(self):
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        start = time.perf_counter()
        with torch.inference_mode():
            for iteration, data_dict in enumerate(self.test_loader, start=1):
                data_dict = to_cuda(data_dict, non_blocking=True)
                output_dict = self.test_step(iteration, data_dict)
                self.after_test_step(iteration, data_dict, output_dict, {})
                if iteration % 100 == 0 or iteration == self.expected_pairs:
                    elapsed = time.perf_counter() - start
                    self.logger.critical(
                        f"Processed {iteration}/{self.expected_pairs} pairs "
                        f"in {elapsed:.1f}s."
                    )
        self.after_test_epoch()
        return {}

    def _get_scene_ground_truth(self, scene_name: str):
        if scene_name not in self.scene_ground_truth:
            gt_root = (
                Path(self.cfg.data.metadata_root)
                / "benchmarks"
                / self.args.benchmark
                / scene_name
            )
            num_fragments = get_num_fragments(scene_name)
            indices, logs, infos = get_gt_logs_and_infos(str(gt_root), num_fragments)
            self.scene_ground_truth[scene_name] = {
                "indices": indices,
                "logs": logs,
                "infos": infos,
                "num_gt_pairs": int((indices != -1).sum()),
                "scene_abbr": get_scene_abbr(scene_name),
            }
        return self.scene_ground_truth[scene_name]

    def test_step(self, iteration, data_dict):
        del iteration
        torch.cuda.synchronize()
        start = time.perf_counter()
        output_dict = self.model(data_dict, compute_gt=True, estimate_transform=False)
        torch.cuda.synchronize()
        self.backbone_times_ms.append((time.perf_counter() - start) * 1000.0)

        budget_outputs = {}
        matching_scores = output_dict["matching_scores"]
        for budget in self.budgets:
            self.model.fine_matching.num_hypotheses = budget
            torch.cuda.synchronize()
            start = time.perf_counter()
            registration = self.model._empty_registration(output_dict["ref_points"])
            succeeded = False
            if matching_scores.shape[0] > 0:
                (
                    ref_corr_points,
                    src_corr_points,
                    corr_scores,
                    estimated_transform,
                    hypotheses,
                    _,
                    _,
                ) = self.model.fine_matching(
                    output_dict["ref_node_corr_knn_points"],
                    output_dict["src_node_corr_knn_points"],
                    output_dict["re_ref_node_corr_knn_feats"],
                    output_dict["re_src_node_corr_knn_feats"],
                    output_dict["ref_node_corr_knn_masks"],
                    output_dict["src_node_corr_knn_masks"],
                    matching_scores,
                )
                succeeded = bool(
                    corr_scores.numel() > 0
                    and hypotheses.shape[0] > 0
                    and torch.isfinite(estimated_transform).all().item()
                )
                if succeeded:
                    registration.update(
                        {
                            "ref_corr_points": ref_corr_points,
                            "src_corr_points": src_corr_points,
                            "corr_scores": corr_scores,
                            "estimated_transform": estimated_transform,
                        }
                    )
            torch.cuda.synchronize()
            self.fhp_times_ms[budget].append((time.perf_counter() - start) * 1000.0)
            registration["succeeded"] = succeeded
            budget_outputs[budget] = registration
        output_dict["budget_outputs"] = budget_outputs
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        del iteration, data_dict, output_dict
        return {}

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        del iteration, output_dict, result_dict
        return (
            f"{data_dict['scene_name']}, id0={data_dict['ref_frame']}, "
            f"id1={data_dict['src_frame']}"
        )

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        del result_dict
        scene_name = data_dict["scene_name"]
        ref_frame = int(data_dict["ref_frame"])
        src_frame = int(data_dict["src_frame"])
        ground_truth = self._get_scene_ground_truth(scene_name)
        gt_index = int(ground_truth["indices"][ref_frame, src_frame])
        official_pair = gt_index != -1
        for budget in self.budgets:
            registration = output_dict["budget_outputs"][budget]
            succeeded = bool(registration["succeeded"])
            accepted = False
            official_rmse = None
            if official_pair and succeeded:
                estimated = np.asarray(
                    release_cuda(registration["estimated_transform"]), dtype=np.float64
                )
                gt_transform = ground_truth["logs"][gt_index]["transform"]
                covariance = ground_truth["infos"][gt_index]["covariance"]
                registration_error = compute_transform_error(
                    gt_transform, covariance, estimated
                )
                if np.isfinite(registration_error):
                    official_rmse = float(np.sqrt(registration_error))
                    accepted = registration_error <= self.cfg.eval.rmse_threshold**2

            self.rows[budget].append(
                {
                    "scene": scene_name,
                    "scene_abbr": ground_truth["scene_abbr"],
                    "ref_frame": ref_frame,
                    "src_frame": src_frame,
                    "pair_key": f"{scene_name}/{ref_frame}_{src_frame}",
                    "official_gt_pair": int(official_pair),
                    "estimation_succeeded": int(succeeded),
                    "accepted": int(accepted),
                    "official_RMSE": "" if official_rmse is None else official_rmse,
                    "shared_backbone_time_ms": self.backbone_times_ms[-1],
                    "fhp_time_ms": self.fhp_times_ms[budget][-1],
                    "pair_iteration": iteration,
                }
            )

    def after_test_epoch(self):
        for budget in self.budgets:
            rows = self.rows[budget]
            scene_results = {}
            for scene_name, ground_truth in self.scene_ground_truth.items():
                scene_rows = [row for row in rows if row["scene"] == scene_name]
                official_rows = [row for row in scene_rows if row["official_gt_pair"] == 1]
                denominator = (
                    len(official_rows)
                    if self.args.pairs_per_scene is not None or self.args.max_pairs is not None
                    else ground_truth["num_gt_pairs"]
                )
                scene_results[ground_truth["scene_abbr"]] = {
                    "RR": (
                        sum(row["accepted"] for row in official_rows)
                        / denominator
                        if denominator
                        else 0.0
                    ),
                    "num_gt_pairs": denominator,
                    "num_evaluated_official_pairs": len(official_rows),
                    "num_accepted": int(sum(row["accepted"] for row in official_rows)),
                }

            overall = {"RR": _mean(scene["RR"] for scene in scene_results.values())}
            summary = {
                "code_version": getattr(self.cfg, "code_version", None),
                "variant": self.cfg.variant,
                "seed": self.cfg.seed,
                "benchmark": self.args.benchmark,
                "method": "fhp",
                "num_hypotheses": budget,
                "registration_protocol": "official_covariance_rr",
                "expected_pairs": self.expected_pairs,
                "evaluated_pairs": len(rows),
                "partial_dataset": (
                    self.args.max_pairs is not None or self.args.pairs_per_scene is not None
                ),
                "pairs_per_scene": self.args.pairs_per_scene,
                "scene_stratified_diagnostic": self.args.pairs_per_scene is not None,
                "snapshot": str(Path(self.args.snapshot).resolve()),
                "shared_backbone_budget_sweep": True,
                "median_shared_backbone_time_ms": float(
                    np.median(self.backbone_times_ms)
                ),
                "median_fhp_time_ms": float(np.median(self.fhp_times_ms[budget])),
                "overall": overall,
                "scenes": scene_results,
            }
            output_root = self.output_roots[budget]
            with (output_root / "pairs_fhp.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            _write_json(output_root / "summary_fhp.json", summary)
            self.logger.critical(
                f"Budget={budget}: RR={overall['RR']:.4f}, output={output_root}"
            )


def main():
    parser = make_parser()
    known_args, _ = parser.parse_known_args()
    cfg = make_cfg(known_args)
    HypothesisBudgetTester(cfg, parser).run()


if __name__ == "__main__":
    main()
