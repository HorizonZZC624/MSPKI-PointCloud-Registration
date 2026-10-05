from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os.path as osp
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    write_log_file,
)
from pareconv.engine import Logger
from pareconv.modules.registration import weighted_procrustes
from pareconv.utils.registration import compute_registration_error

from config import get_benchmark_tag, make_cfg, make_experiment_parser


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description="Official and diagnostic evaluation of saved PARE-Net outputs.",
    )
    parser.add_argument(
        "--benchmark",
        default="3DMatch",
        choices=["3DMatch", "3DLoMatch"],
    )
    parser.add_argument(
        "--method",
        default="fhp",
        choices=["fhp", "ransac", "svd"],
    )
    parser.add_argument("--num_corr", type=int, default=None)
    parser.add_argument("--topk_values", type=int, nargs="*", default=[50, 100, 250])
    parser.add_argument(
        "--coarse_overlap_thresholds",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.3],
    )
    parser.add_argument(
        "--overlap_bins",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.2, 0.3, 0.5, 1.01],
    )
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.median(values)) if values else 0.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(np.asarray(value).item())
    except (ValueError, TypeError):
        return default
    return result if math.isfinite(result) else default


def _is_valid_rigid_transform(transform: np.ndarray) -> bool:
    transform = np.asarray(transform)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4, rtol=0):
        return False
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2, rtol=0):
        return False
    return abs(float(np.linalg.det(rotation)) - 1.0) < 1e-2


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _method_tag(method: str, num_corr: Optional[int]) -> str:
    return method if num_corr is None else f"{method}_n{int(num_corr)}"


def _load_and_validate_manifest(
    features_root: str,
    cfg,
    benchmark: str,
    allow_partial: bool,
) -> Dict:
    manifest_file = osp.join(features_root, "manifest.json")
    if not osp.isfile(manifest_file):
        raise FileNotFoundError(
            f"Missing {manifest_file}. Run paper_test.py before evaluation."
        )
    with open(manifest_file, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    expected = {
        "code_version": getattr(cfg, "code_version", None),
        "variant": cfg.variant,
        "seed": cfg.seed,
        "re_feature_source": cfg.fine_matching.re_feature_source,
        "benchmark": benchmark,
        "rotation_mode": cfg.test.rotation_mode,
        "rotation_seed": cfg.test.rotation_seed,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Feature manifest/config mismatch: {mismatches}")
    if not manifest.get("completed", False):
        raise RuntimeError("Feature extraction is marked incomplete.")
    if manifest.get("partial_dataset", False) and not allow_partial:
        raise RuntimeError(
            "This extraction used --max_pairs and is not valid for paper statistics. "
            "Pass --allow_partial only for a smoke test."
        )

    expected_pairs = int(manifest.get("expected_pairs", -1))
    saved_pairs = int(manifest.get("saved_pairs", -1))
    actual_pairs = len(glob.glob(osp.join(features_root, "*", "*.npz")))
    if expected_pairs < 0 or saved_pairs != expected_pairs or actual_pairs != expected_pairs:
        raise RuntimeError(
            "Feature count mismatch: "
            f"expected={expected_pairs}, manifest_saved={saved_pairs}, actual={actual_pairs}."
        )
    return manifest


def _estimate_transform(
    method: str,
    num_corr: Optional[int],
    cfg,
    data_dict,
    ref_points: np.ndarray,
    src_points: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    identity = np.eye(4, dtype=np.float32)
    if num_corr is not None and scores.shape[0] > int(num_corr):
        selected = np.argsort(-scores)[: int(num_corr)]
        ref_points = ref_points[selected]
        src_points = src_points[selected]
        scores = scores[selected]

    if method == "fhp":
        if num_corr is not None:
            raise ValueError("num_corr cannot be used with FHP saved transforms.")
        succeeded = bool(np.asarray(data_dict["registration_succeeded"]).item())
        transform = np.asarray(data_dict["estimated_transform"])
        succeeded = succeeded and ref_points.shape[0] > 0 and _is_valid_rigid_transform(transform)
        return (transform if succeeded else identity), succeeded

    if ref_points.shape[0] < 3 or src_points.shape[0] < 3:
        return identity, False

    if method == "ransac":
        from pareconv.utils.open3d import registration_with_ransac_from_correspondences

        try:
            import open3d as o3d

            try:
                o3d.utility.random.seed(int(cfg.seed))
            except AttributeError:
                pass
            transform = registration_with_ransac_from_correspondences(
                src_points,
                ref_points,
                distance_threshold=cfg.ransac.distance_threshold,
                ransac_n=cfg.ransac.num_points,
                num_iterations=cfg.ransac.num_iterations,
            )
        except (RuntimeError, ValueError):
            return identity, False
        transform = np.asarray(transform)
        succeeded = _is_valid_rigid_transform(transform)
        return (transform if succeeded else identity), succeeded

    if method == "svd":
        if (
            scores.shape[0] != src_points.shape[0]
            or not np.isfinite(scores).all()
            or not np.any(scores > 0)
        ):
            return identity, False
        centered = src_points - src_points.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 2:
            return identity, False
        with torch.no_grad():
            transform = weighted_procrustes(
                torch.from_numpy(src_points).float(),
                torch.from_numpy(ref_points).float(),
                torch.from_numpy(scores).float(),
                return_transform=True,
            ).cpu().numpy()
        succeeded = _is_valid_rigid_transform(transform)
        return (transform if succeeded else identity), succeeded

    raise ValueError(f"Unsupported method: {method}")


def _coarse_diagnostics(
    ref_pred: np.ndarray,
    src_pred: np.ndarray,
    gt_indices: np.ndarray,
    gt_overlaps: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[str, float]:
    ref_pred = np.asarray(ref_pred, dtype=np.int64).reshape(-1)
    src_pred = np.asarray(src_pred, dtype=np.int64).reshape(-1)
    gt_indices = np.asarray(gt_indices, dtype=np.int64).reshape(-1, 2)
    gt_overlaps = np.asarray(gt_overlaps, dtype=np.float64).reshape(-1)

    overlap_map = {
        (int(pair[0]), int(pair[1])): float(overlap)
        for pair, overlap in zip(gt_indices, gt_overlaps)
    }
    predicted_overlaps = np.asarray(
        [overlap_map.get((int(r), int(s)), 0.0) for r, s in zip(ref_pred, src_pred)],
        dtype=np.float64,
    )

    result: Dict[str, float] = {
        "num_coarse_corr": float(ref_pred.shape[0]),
        "mean_predicted_gt_overlap": (
            float(predicted_overlaps.mean()) if predicted_overlaps.size else 0.0
        ),
    }
    for threshold in thresholds:
        tag = str(float(threshold)).replace(".", "p")
        gt_mask = gt_overlaps > float(threshold)
        gt_set = {
            (int(pair[0]), int(pair[1]))
            for pair in gt_indices[gt_mask]
        }
        pred_set = {(int(r), int(s)) for r, s in zip(ref_pred, src_pred)}
        correct = pred_set & gt_set
        result[f"PIR_gt{tag}"] = (
            float(len(correct)) / float(len(pred_set)) if pred_set else 0.0
        )
        result[f"coarse_recall_gt{tag}"] = (
            float(len(correct)) / float(len(gt_set)) if gt_set else 0.0
        )

        gt_ref = {pair[0] for pair in gt_set}
        gt_src = {pair[1] for pair in gt_set}
        correct_ref = {pair[0] for pair in correct}
        correct_src = {pair[1] for pair in correct}
        result[f"ref_coverage_gt{tag}"] = (
            float(len(correct_ref)) / float(len(gt_ref)) if gt_ref else 0.0
        )
        result[f"src_coverage_gt{tag}"] = (
            float(len(correct_src)) / float(len(gt_src)) if gt_src else 0.0
        )
    return result


def _fine_diagnostics(
    ref_corr: np.ndarray,
    src_corr: np.ndarray,
    scores: np.ndarray,
    transform: np.ndarray,
    positive_radius: float,
    topk_values: Sequence[int],
) -> Dict[str, float]:
    ref_corr = np.asarray(ref_corr, dtype=np.float64).reshape(-1, 3)
    src_corr = np.asarray(src_corr, dtype=np.float64).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    num_corr = int(ref_corr.shape[0])
    if num_corr == 0:
        result = {
            "num_corr": 0.0,
            "num_inliers": 0.0,
            "IR": 0.0,
            "weighted_IR": 0.0,
            "mean_residual": 0.0,
            "inlier_spread_trace": 0.0,
            "inlier_spread_minmax_ratio": 0.0,
            "inlier_geometry_rank": 0.0,
        }
        for topk in topk_values:
            result[f"IR_top{int(topk)}"] = 0.0
        return result

    aligned_src = _transform_points(src_corr, transform)
    distances = np.linalg.norm(ref_corr - aligned_src, axis=1)
    inlier_mask = distances < float(positive_radius)
    num_inliers = int(inlier_mask.sum())
    weights = np.clip(scores, 0.0, None)
    weighted_ir = (
        float(np.sum(weights * inlier_mask.astype(np.float64)) / np.sum(weights))
        if np.sum(weights) > 0
        else 0.0
    )

    result = {
        "num_corr": float(num_corr),
        "num_inliers": float(num_inliers),
        "IR": float(inlier_mask.mean()),
        "weighted_IR": weighted_ir,
        "mean_residual": float(distances.mean()),
    }
    order = np.argsort(-scores)
    for topk in topk_values:
        count = min(int(topk), num_corr)
        result[f"IR_top{int(topk)}"] = (
            float(inlier_mask[order[:count]].mean()) if count > 0 else 0.0
        )

    inlier_ref = ref_corr[inlier_mask]
    if inlier_ref.shape[0] >= 3:
        centered = inlier_ref - inlier_ref.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(inlier_ref.shape[0] - 1, 1)
        eigvals = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        maximum = float(eigvals[-1]) if eigvals.size else 0.0
        result["inlier_spread_trace"] = float(eigvals.sum())
        result["inlier_spread_minmax_ratio"] = (
            float(eigvals[0] / maximum) if maximum > 0 else 0.0
        )
        result["inlier_geometry_rank"] = float(np.linalg.matrix_rank(centered))
    else:
        result["inlier_spread_trace"] = 0.0
        result["inlier_spread_minmax_ratio"] = 0.0
        result["inlier_geometry_rank"] = float(inlier_ref.shape[0])
    return result


def _aggregate_rows(rows: Sequence[Dict], keys: Sequence[str]) -> Dict[str, float]:
    return {key: _mean(row.get(key, 0.0) for row in rows) for key in keys}


def _overlap_bin_summary(rows: Sequence[Dict], bins: Sequence[float]) -> List[Dict]:
    summaries: List[Dict] = []
    for low, high in zip(bins[:-1], bins[1:]):
        selected = [
            row
            for row in rows
            if float(low) <= float(row.get("pcd_overlap", 0.0)) < float(high)
            and int(row.get("official_gt_pair", 1)) == 1
        ]
        summaries.append(
            {
                "overlap_low": float(low),
                "overlap_high": float(high),
                "num_pairs": len(selected),
                "RR": _mean(row.get("accepted", 0.0) for row in selected),
                "IR": _mean(row.get("IR", 0.0) for row in selected),
                "weighted_IR": _mean(row.get("weighted_IR", 0.0) for row in selected),
                "FMR": _mean(row.get("FMR", 0.0) for row in selected),
                "mean_RRE": _mean(
                    row.get("RRE", 0.0)
                    for row in selected
                    if row.get("estimation_succeeded", 0) == 1
                ),
                "mean_RTE": _mean(
                    row.get("RTE", 0.0)
                    for row in selected
                    if row.get("estimation_succeeded", 0) == 1
                ),
            }
        )
    return summaries


def _write_csv(rows: Sequence[Dict], file_name: str) -> None:
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(file_name, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args, cfg, logger: Logger) -> Dict:
    if args.method == "fhp" and args.num_corr is not None:
        raise ValueError("--num_corr is incompatible with --method fhp.")
    if args.num_corr is not None and int(args.num_corr) < 1:
        raise ValueError("--num_corr must be positive.")

    benchmark_tag = get_benchmark_tag(cfg, args.benchmark)
    method_tag = _method_tag(args.method, args.num_corr)
    features_root = osp.join(cfg.feature_dir, benchmark_tag)
    manifest = _load_and_validate_manifest(
        features_root,
        cfg,
        args.benchmark,
        allow_partial=args.allow_partial,
    )
    logger.info(f"Feature manifest: {json.dumps(manifest, indent=2)}")

    scene_roots = sorted(
        path for path in glob.glob(osp.join(features_root, "*")) if osp.isdir(path)
    )
    if not scene_roots:
        raise FileNotFoundError(f"No scene outputs found under {features_root}.")

    scene_results: Dict[str, Dict] = {}
    all_pair_rows: List[Dict] = []
    official_scene_keys = [
        "PIR",
        "FMR",
        "IR",
        "OV",
        "RR",
        "mean_RRE",
        "mean_RTE",
        "median_RRE",
        "median_RTE",
    ]
    official_aggregate: Dict[str, List[float]] = {key: [] for key in official_scene_keys}

    for scene_root in scene_roots:
        scene_name = osp.basename(scene_root)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)

        gt_indices = gt_logs = gt_infos = None
        if cfg.test.rotation_mode == "none":
            gt_root = osp.join(
                cfg.data.metadata_root,
                "benchmarks",
                args.benchmark,
                scene_name,
            )
            gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_root, num_fragments)
            num_gt_pairs = int((gt_indices != -1).sum())
        else:
            num_gt_pairs = 0

        file_names = sorted(
            glob.glob(osp.join(scene_root, "*.npz")),
            key=lambda path: [
                int(item) for item in osp.basename(path).split(".")[0].split("_")
            ],
        )
        if not file_names:
            raise RuntimeError(f"No .npz files found for scene {scene_name}.")

        scene_rows: List[Dict] = []
        estimated_transforms = []
        failed_pairs = []
        num_pred_gt_pairs = 0

        for file_name in file_names:
            ref_frame, src_frame = [
                int(item) for item in osp.basename(file_name).split(".")[0].split("_")
            ]
            with np.load(file_name) as data:
                pred_ref = np.asarray(data["ref_node_corr_indices"])
                pred_src = np.asarray(data["src_node_corr_indices"])
                gt_node_indices = np.asarray(data["gt_node_corr_indices"])
                gt_node_overlaps = np.asarray(data["gt_node_corr_overlaps"])
                saved_transform = np.asarray(data["transform"])
                pcd_overlap = _safe_float(data["overlap"])
                ref_corr = np.asarray(data["ref_corr_points"])
                src_corr = np.asarray(data["src_corr_points"])
                corr_scores = np.asarray(data["corr_scores"])
                forward_time_ms = _safe_float(
                    data["forward_time_ms"] if "forward_time_ms" in data.files else 0.0
                )

                coarse = _coarse_diagnostics(
                    pred_ref,
                    pred_src,
                    gt_node_indices,
                    gt_node_overlaps,
                    args.coarse_overlap_thresholds,
                )
                diag_ref_corr = ref_corr
                diag_src_corr = src_corr
                diag_scores = corr_scores
                if (
                    args.method != "fhp"
                    and args.num_corr is not None
                    and corr_scores.shape[0] > int(args.num_corr)
                ):
                    selected = np.argsort(-corr_scores)[: int(args.num_corr)]
                    diag_ref_corr = ref_corr[selected]
                    diag_src_corr = src_corr[selected]
                    diag_scores = corr_scores[selected]
                fine = _fine_diagnostics(
                    diag_ref_corr,
                    diag_src_corr,
                    diag_scores,
                    saved_transform,
                    cfg.eval.acceptance_radius,
                    args.topk_values,
                )
                estimated_transform, estimation_succeeded = _estimate_transform(
                    args.method,
                    args.num_corr,
                    cfg,
                    data,
                    ref_corr,
                    src_corr,
                    corr_scores,
                )

                if estimation_succeeded:
                    estimated_transforms.append(
                        {
                            "test_pair": [ref_frame, src_frame],
                            "num_fragments": num_fragments,
                            "transform": estimated_transform,
                        }
                    )
                else:
                    failed_pairs.append([ref_frame, src_frame])

                official_gt_pair = True
                accepted = False
                registration_error = None
                rre = rte = None
                if cfg.test.rotation_mode == "none":
                    gt_index = int(gt_indices[ref_frame, src_frame])
                    official_gt_pair = gt_index != -1
                    if official_gt_pair and estimation_succeeded:
                        num_pred_gt_pairs += 1
                        gt_transform = gt_logs[gt_index]["transform"]
                        covariance = gt_infos[gt_index]["covariance"]
                        registration_error = compute_transform_error(
                            gt_transform, covariance, estimated_transform
                        )
                        accepted = bool(
                            np.isfinite(registration_error)
                            and registration_error <= cfg.eval.rmse_threshold**2
                        )
                        rre, rte = compute_registration_error(
                            gt_transform, estimated_transform
                        )
                else:
                    if estimation_succeeded:
                        rre, rte = compute_registration_error(
                            saved_transform, estimated_transform
                        )
                        accepted = bool(
                            np.isfinite(rre)
                            and np.isfinite(rte)
                            and rre <= cfg.eval.rre_threshold
                            and rte <= cfg.eval.rte_threshold
                        )

                row = {
                    "scene": scene_name,
                    "scene_abbr": scene_abbr,
                    "ref_frame": ref_frame,
                    "src_frame": src_frame,
                    "pair_key": f"{scene_name}/{ref_frame}_{src_frame}",
                    "official_gt_pair": int(official_gt_pair),
                    "pcd_overlap": pcd_overlap,
                    "FMR": float(fine["IR"] >= cfg.eval.inlier_ratio_threshold),
                    "estimation_succeeded": int(estimation_succeeded),
                    "accepted": int(accepted),
                    "RRE": float(rre) if rre is not None and np.isfinite(rre) else "",
                    "RTE": float(rte) if rte is not None and np.isfinite(rte) else "",
                    "official_RMSE": (
                        float(np.sqrt(registration_error))
                        if registration_error is not None and np.isfinite(registration_error)
                        else ""
                    ),
                    "forward_time_ms": forward_time_ms,
                }
                row.update(coarse)
                row.update(fine)
                scene_rows.append(row)
                all_pair_rows.append(row)

                if args.verbose:
                    logger.info(
                        f"{row['pair_key']}: OV={pcd_overlap:.3f}, "
                        f"PIR={row.get('PIR_gt0p0', 0.0):.3f}, "
                        f"IR={row['IR']:.3f}, WIR={row['weighted_IR']:.3f}, "
                        f"N={int(row['num_corr'])}, inliers={int(row['num_inliers'])}, "
                        f"success={int(estimation_succeeded)}, accepted={int(accepted)}"
                    )

        est_log = osp.join(
            cfg.registration_dir,
            benchmark_tag,
            scene_name,
            f"{method_tag}.log",
        )
        write_log_file(est_log, estimated_transforms)

        official_rows = [row for row in scene_rows if row["official_gt_pair"] == 1]
        accepted_rres = [
            float(row["RRE"])
            for row in official_rows
            if row["accepted"] == 1 and row["RRE"] != ""
        ]
        accepted_rtes = [
            float(row["RTE"])
            for row in official_rows
            if row["accepted"] == 1 and row["RTE"] != ""
        ]
        if cfg.test.rotation_mode == "none":
            rr = (
                float(sum(row["accepted"] for row in official_rows)) / num_gt_pairs
                if num_gt_pairs > 0
                else 0.0
            )
            precision = (
                float(sum(row["accepted"] for row in official_rows)) / num_pred_gt_pairs
                if num_pred_gt_pairs > 0
                else 0.0
            )
        else:
            num_gt_pairs = len(official_rows)
            num_pred_gt_pairs = sum(row["estimation_succeeded"] for row in official_rows)
            rr = _mean(row["accepted"] for row in official_rows)
            precision = rr

        scene_result = {
            "PIR": _mean(row.get("PIR_gt0p0", 0.0) for row in scene_rows),
            "FMR": _mean(row["FMR"] for row in scene_rows),
            "IR": _mean(row["IR"] for row in scene_rows),
            "OV": _mean(row["pcd_overlap"] for row in scene_rows),
            "RR": rr,
            "registration_precision": precision,
            "mean_RRE": _mean(accepted_rres),
            "mean_RTE": _mean(accepted_rtes),
            "median_RRE": _median(accepted_rres),
            "median_RTE": _median(accepted_rtes),
            "num_gt_pairs": int(num_gt_pairs),
            "num_pred_gt_pairs": int(num_pred_gt_pairs),
            "num_accepted": int(sum(row["accepted"] for row in official_rows)),
            "num_failed_estimations": int(
                sum(1 - row["estimation_succeeded"] for row in scene_rows)
            ),
            "failed_pairs": failed_pairs,
        }
        scene_results[scene_abbr] = scene_result
        for key in official_scene_keys:
            official_aggregate[key].append(scene_result[key])

        logger.info(
            f"{scene_abbr}: PIR={scene_result['PIR']:.3f}, "
            f"FMR={scene_result['FMR']:.3f}, IR={scene_result['IR']:.3f}, "
            f"RR={scene_result['RR']:.3f}, "
            f"mean_RRE={scene_result['mean_RRE']:.3f}, "
            f"mean_RTE={scene_result['mean_RTE']:.3f}"
        )

    overall = {key: _mean(values) for key, values in official_aggregate.items()}
    diagnostic_keys = [
        "PIR_gt0p0",
        "PIR_gt0p1",
        "PIR_gt0p3",
        "coarse_recall_gt0p0",
        "coarse_recall_gt0p1",
        "coarse_recall_gt0p3",
        "ref_coverage_gt0p1",
        "src_coverage_gt0p1",
        "mean_predicted_gt_overlap",
        "num_corr",
        "num_inliers",
        "IR",
        "weighted_IR",
        "mean_residual",
        "inlier_spread_trace",
        "inlier_spread_minmax_ratio",
        "forward_time_ms",
    ] + [f"IR_top{int(topk)}" for topk in args.topk_values]
    pair_mean = _aggregate_rows(all_pair_rows, diagnostic_keys)
    pair_mean["FMR"] = _mean(row["FMR"] for row in all_pair_rows)
    pair_mean["pairwise_success_rate"] = _mean(
        row["accepted"] for row in all_pair_rows if row["official_gt_pair"] == 1
    )

    overlap_summary = _overlap_bin_summary(all_pair_rows, args.overlap_bins)

    logger.critical(
        f"Overall official: PIR={overall['PIR']:.3f}, FMR={overall['FMR']:.3f}, "
        f"IR={overall['IR']:.3f}, RR={overall['RR']:.3f}, "
        f"mean_RRE={overall['mean_RRE']:.3f}, mean_RTE={overall['mean_RTE']:.3f}"
    )
    logger.critical(
        f"Diagnostic: WIR={pair_mean.get('weighted_IR', 0.0):.3f}, "
        f"num_corr={pair_mean.get('num_corr', 0.0):.1f}, "
        f"num_inliers={pair_mean.get('num_inliers', 0.0):.1f}, "
        f"forward_ms={pair_mean.get('forward_time_ms', 0.0):.2f}"
    )

    summary = {
        "code_version": getattr(cfg, "code_version", None),
        "variant": cfg.variant,
        "seed": cfg.seed,
        "re_feature_source": cfg.fine_matching.re_feature_source,
        "benchmark": args.benchmark,
        "benchmark_tag": benchmark_tag,
        "rotation_mode": cfg.test.rotation_mode,
        "rotation_seed": cfg.test.rotation_seed,
        "condition_name": manifest.get("condition_name", ""),
        "keep_ratio": manifest.get("keep_ratio", 1.0),
        "noise_std": manifest.get("noise_std", 0.0),
        "registration_protocol": (
            "official_covariance_rr"
            if cfg.test.rotation_mode == "none"
            else "rre_rte_success_rate"
        ),
        "method": args.method,
        "method_tag": method_tag,
        "num_corr": args.num_corr,
        "feature_manifest": manifest,
        "overall": overall,
        "pair_mean_diagnostics": pair_mean,
        "overlap_bins": overlap_summary,
        "scenes": scene_results,
        "num_pair_rows": len(all_pair_rows),
    }

    output_dir = osp.join(cfg.registration_dir, benchmark_tag)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_file = osp.join(output_dir, f"summary_{method_tag}.json")
    pair_csv = osp.join(output_dir, f"pairs_{method_tag}.csv")
    overlap_csv = osp.join(output_dir, f"overlap_bins_{method_tag}.csv")
    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    _write_csv(all_pair_rows, pair_csv)
    _write_csv(overlap_summary, overlap_csv)
    logger.critical(f"Summary saved to {summary_file}")
    logger.critical(f"Pair diagnostics saved to {pair_csv}")
    return summary


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg(args)
    log_file = osp.join(
        cfg.log_dir,
        f"paper-eval-{time.strftime('%Y%m%d-%H%M%S')}.log",
    )
    logger = Logger(log_file=log_file)
    logger.info("Command executed: " + " ".join(sys.argv))
    logger.info("Configs:\n" + json.dumps(cfg, indent=4))
    evaluate(args, cfg, logger)


if __name__ == "__main__":
    main()
