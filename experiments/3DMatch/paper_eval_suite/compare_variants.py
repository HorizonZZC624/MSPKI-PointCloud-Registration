from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


VARIANT_ORDER = ("baseline", "spsa", "mspki", "full")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate PARE-Net paper-evaluation results across variants."
    )
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--re_feature_source", default="official")
    parser.add_argument("--method_tag", default="fhp")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=7351)
    return parser


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(rows: Sequence[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    lower = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * lower)


def _paired_bootstrap(
    full_success: np.ndarray,
    other_success: np.ndarray,
    num_samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    if full_success.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    differences = full_success.astype(np.float64) - other_success.astype(np.float64)
    point = float(differences.mean())
    sample_means = np.empty(int(num_samples), dtype=np.float64)
    for index in range(int(num_samples)):
        selected = rng.integers(0, differences.size, size=differences.size)
        sample_means[index] = differences[selected].mean()
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return point, float(low), float(high)


def _summary_pattern(
    condition_root: str,
    variant: str,
    re_feature_source: str,
    seed: int,
    method_tag: str,
) -> str:
    return os.path.join(
        condition_root,
        variant,
        f"re_{re_feature_source}",
        f"seed_{seed}",
        "registration",
        "*",
        f"summary_{method_tag}.json",
    )


def _find_single(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Expected one file for pattern {pattern!r}, found {matches}")
    return matches[0]


def _flatten_summary(summary: Dict, condition_dir: str, summary_path: str) -> Dict:
    overall = summary.get("overall", {})
    diagnostics = summary.get("pair_mean_diagnostics", {})
    manifest = summary.get("feature_manifest", {})
    row = {
        "condition_dir": condition_dir,
        "condition_name": summary.get("condition_name") or condition_dir,
        "variant": summary.get("variant"),
        "seed": summary.get("seed"),
        "benchmark": summary.get("benchmark"),
        "benchmark_tag": summary.get("benchmark_tag"),
        "rotation_mode": summary.get("rotation_mode"),
        "keep_ratio": summary.get("keep_ratio", 1.0),
        "noise_std": summary.get("noise_std", 0.0),
        "method": summary.get("method"),
        "method_tag": summary.get("method_tag"),
        "RR": overall.get("RR", 0.0),
        "FMR": overall.get("FMR", 0.0),
        "IR": overall.get("IR", 0.0),
        "PIR": overall.get("PIR", 0.0),
        "mean_RRE": overall.get("mean_RRE", 0.0),
        "mean_RTE": overall.get("mean_RTE", 0.0),
        "weighted_IR": diagnostics.get("weighted_IR", 0.0),
        "IR_top50": diagnostics.get("IR_top50", 0.0),
        "IR_top100": diagnostics.get("IR_top100", 0.0),
        "IR_top250": diagnostics.get("IR_top250", 0.0),
        "num_corr": diagnostics.get("num_corr", 0.0),
        "num_inliers": diagnostics.get("num_inliers", 0.0),
        "PIR_gt0p1": diagnostics.get("PIR_gt0p1", 0.0),
        "coarse_recall_gt0p1": diagnostics.get("coarse_recall_gt0p1", 0.0),
        "forward_time_ms": diagnostics.get(
            "forward_time_ms", manifest.get("mean_forward_time_ms", 0.0)
        ),
        "parameters": manifest.get("parameters", 0),
        "peak_cuda_memory_mb": manifest.get("peak_cuda_memory_mb", 0.0),
        "summary_path": summary_path,
    }
    return row


def _load_pair_map(summary_path: str, method_tag: str) -> Dict[str, Dict[str, str]]:
    pair_path = os.path.join(os.path.dirname(summary_path), f"pairs_{method_tag}.csv")
    if not os.path.isfile(pair_path):
        return {}
    rows = _read_csv(pair_path)
    return {row["pair_key"]: row for row in rows if row.get("official_gt_pair", "1") == "1"}


def aggregate(args) -> Dict:
    results_root = os.path.abspath(os.path.expanduser(args.results_root))
    condition_roots = sorted(
        path for path in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(path)
    )
    all_rows: List[Dict] = []
    summary_index: Dict[Tuple[str, str], str] = {}

    for condition_root in condition_roots:
        condition_dir = os.path.basename(condition_root)
        for variant in VARIANT_ORDER:
            pattern = _summary_pattern(
                condition_root,
                variant,
                args.re_feature_source,
                args.seed,
                args.method_tag,
            )
            summary_path = _find_single(pattern)
            if summary_path is None:
                continue
            summary = _read_json(summary_path)
            all_rows.append(_flatten_summary(summary, condition_dir, summary_path))
            summary_index[(condition_dir, variant)] = summary_path

    if not all_rows:
        raise FileNotFoundError(
            f"No summary_{args.method_tag}.json files found under {results_root}."
        )

    comparisons: List[Dict] = []
    ranking_rows: List[Dict] = []
    readiness_conditions: Dict[str, Dict] = {}

    conditions = sorted({row["condition_dir"] for row in all_rows})
    for condition in conditions:
        rows = [row for row in all_rows if row["condition_dir"] == condition]
        rows_by_variant = {row["variant"]: row for row in rows}
        ranking = sorted(rows, key=lambda row: _as_float(row["RR"]), reverse=True)
        for rank, row in enumerate(ranking, start=1):
            ranking_rows.append(
                {
                    "condition_dir": condition,
                    "rank": rank,
                    "variant": row["variant"],
                    "RR": row["RR"],
                    "FMR": row["FMR"],
                    "IR": row["IR"],
                    "weighted_IR": row["weighted_IR"],
                    "mean_RRE": row["mean_RRE"],
                    "mean_RTE": row["mean_RTE"],
                }
            )

        full_row = rows_by_variant.get("full")
        if full_row is None:
            continue
        condition_report = {
            "full_rr": _as_float(full_row["RR"]),
            "full_rank": next(
                (index + 1 for index, row in enumerate(ranking) if row["variant"] == "full"),
                None,
            ),
            "full_is_best_rr": bool(ranking and ranking[0]["variant"] == "full"),
            "comparisons": {},
        }

        full_pair_map = _load_pair_map(
            summary_index[(condition, "full")], args.method_tag
        )
        for other_variant in ("baseline", "spsa", "pki"):
            other_row = rows_by_variant.get(other_variant)
            if other_row is None:
                continue
            comparison = {
                "condition_dir": condition,
                "full_vs": other_variant,
                "delta_RR": _as_float(full_row["RR"]) - _as_float(other_row["RR"]),
                "delta_FMR": _as_float(full_row["FMR"]) - _as_float(other_row["FMR"]),
                "delta_IR": _as_float(full_row["IR"]) - _as_float(other_row["IR"]),
                "delta_weighted_IR": _as_float(full_row["weighted_IR"])
                - _as_float(other_row["weighted_IR"]),
                "delta_RRE": _as_float(full_row["mean_RRE"])
                - _as_float(other_row["mean_RRE"]),
                "delta_RTE": _as_float(full_row["mean_RTE"])
                - _as_float(other_row["mean_RTE"]),
                "delta_forward_time_ms": _as_float(full_row["forward_time_ms"])
                - _as_float(other_row["forward_time_ms"]),
            }

            other_pair_map = _load_pair_map(
                summary_index[(condition, other_variant)], args.method_tag
            )
            common = sorted(set(full_pair_map) & set(other_pair_map))
            if common:
                full_success = np.asarray(
                    [int(float(full_pair_map[key].get("accepted", 0))) for key in common],
                    dtype=np.int64,
                )
                other_success = np.asarray(
                    [int(float(other_pair_map[key].get("accepted", 0))) for key in common],
                    dtype=np.int64,
                )
                full_only = int(np.sum((full_success == 1) & (other_success == 0)))
                other_only = int(np.sum((full_success == 0) & (other_success == 1)))
                point, low, high = _paired_bootstrap(
                    full_success,
                    other_success,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                )
                comparison.update(
                    {
                        "matched_pairs": len(common),
                        "full_only_success": full_only,
                        "other_only_success": other_only,
                        "mcnemar_exact_p": _mcnemar_exact_p(full_only, other_only),
                        "paired_delta_success": point,
                        "paired_delta_95ci_low": low,
                        "paired_delta_95ci_high": high,
                    }
                )
            comparisons.append(comparison)
            condition_report["comparisons"][other_variant] = comparison
        readiness_conditions[condition] = condition_report

    baseline_rows = {row["condition_dir"]: row for row in all_rows if row["variant"] == "baseline"}
    full_rows = {row["condition_dir"]: row for row in all_rows if row["variant"] == "full"}
    runtime_overheads = []
    for condition in sorted(set(baseline_rows) & set(full_rows)):
        baseline_time = _as_float(baseline_rows[condition]["forward_time_ms"])
        full_time = _as_float(full_rows[condition]["forward_time_ms"])
        if baseline_time > 0:
            runtime_overheads.append((full_time - baseline_time) / baseline_time)

    required_core = {
        "standard_3DMatch",
        "standard_3DLoMatch",
        "so3_3DMatch",
        "so3_3DLoMatch",
    }
    available = set(conditions)
    readiness = {
        "single_seed_warning": (
            "These results use one trained seed only. They cannot establish "
            "multi-seed stability or a publication-grade mean±std conclusion."
        ),
        "required_core_conditions": sorted(required_core),
        "missing_core_conditions": sorted(required_core - available),
        "all_core_conditions_available": required_core.issubset(available),
        "full_best_rr_count": sum(
            int(report.get("full_is_best_rr", False))
            for name, report in readiness_conditions.items()
            if name in required_core
        ),
        "core_condition_count": sum(name in available for name in required_core),
        "mean_full_runtime_overhead_fraction": (
            float(np.mean(runtime_overheads)) if runtime_overheads else None
        ),
        "conditions": readiness_conditions,
        "interpretation_rule": (
            "The strongest single-seed evidence is: Full ranks first in official "
            "3DLoMatch and rotated 3DLoMatch RR, Full-only successes exceed the "
            "reverse failures, and the gain is not explained by excessive runtime. "
            "Multi-seed experiments remain necessary before claiming stability."
        ),
    }

    output_dir = os.path.join(results_root, "aggregate")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_csv(all_rows, os.path.join(output_dir, "all_results.csv"))
    _write_csv(ranking_rows, os.path.join(output_dir, "rankings.csv"))
    _write_csv(comparisons, os.path.join(output_dir, "full_pairwise_comparisons.csv"))
    with open(
        os.path.join(output_dir, "paper_readiness_single_seed.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(readiness, file, indent=2, ensure_ascii=False)

    print(f"Aggregated {len(all_rows)} summaries into: {output_dir}")
    print(json.dumps(readiness, indent=2, ensure_ascii=False))
    return readiness


def main() -> None:
    args = make_parser().parse_args()
    aggregate(args)


if __name__ == "__main__":
    main()
