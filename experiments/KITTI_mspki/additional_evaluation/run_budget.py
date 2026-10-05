from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from common import ROOT, SEEDS, METHODS, cache_files, correspondence_metrics, metadata_keys, pair_key, paired_summary, pose_metrics, write_csv, write_json


def parser():
    p = argparse.ArgumentParser(description="KITTI top-score correspondence budgets with seeded RANSAC.")
    p.add_argument("--cache-root", type=Path, default=ROOT / "output/KITTI_mspki")
    p.add_argument("--cache-tag", default="", help="Empty uses the existing full-density seed-level features.")
    p.add_argument("--metadata-root", type=Path, default=ROOT / "data/KITTI/metadata")
    p.add_argument("--output", type=Path, default=ROOT / "output/KITTI_additional/budget")
    p.add_argument("--training-seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--estimator-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 250, 500, 0], help="0 means All.")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-pairs", type=int, default=0, help="Nonzero produces smoke-test outputs only.")
    return p


def main():
    args = parser().parse_args()
    if args.iterations <= 0 or args.max_pairs < 0 or any(x < 0 for x in args.budgets):
        raise ValueError("Iterations must be positive; budgets/max-pairs must be nonnegative.")
    if any(x < 0 for x in args.estimator_seeds):
        raise ValueError("Estimator seeds must be nonnegative.")
    _, expected = metadata_keys(args.metadata_root)
    caches = {}
    for method in METHODS:
        for seed in args.training_seeds:
            folder = args.cache_root / method / "re_official" / f"seed_{seed}" / args.cache_tag / "features"
            files = cache_files(folder, expected)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if not manifest.get("completed") or manifest.get("saved_pairs") != 555:
                raise ValueError(f"Incomplete manifest: {folder}")
            if manifest.get("variant") != method or manifest.get("seed") != seed or manifest.get("re_feature_source") != "official":
                raise ValueError(f"Mismatched method, training seed, or RE source: {folder}")
            if float(manifest.get("density_keep_ratio", 1.0)) != 1.0:
                raise ValueError("The budget experiment requires full-density input caches.")
            caches[(method, seed)] = (files, manifest, folder)
    plan = {"pairs_per_run": args.max_pairs or 555, "training_seeds": args.training_seeds,
            "estimator_seeds": args.estimator_seeds, "budgets": args.budgets,
            "cache_tag": args.cache_tag or "seed-level full-density cache",
            "pose_evaluations": 2 * len(args.training_seeds) * len(args.estimator_seeds) * len(args.budgets) * (args.max_pairs or 555),
            "RANSAC_distance_m": 0.3, "RANSAC_minimal_points": 4,
            "iterations": args.iterations, "confidence": 0.999,
            "IR_radius_m": 1.0, "TR_RRE_strict_deg": 5.0, "TR_RTE_strict_m": 2.0,
            "score_ties": "stable input order", "random_seed_policy": "reset before each pair",
            "training_seed_aggregation": "average estimator repeats within each training seed, then sample SD over training seeds",
            "complete_test_set": args.max_pairs == 0}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    import open3d as o3d
    if not hasattr(o3d.utility, "random"):
        raise RuntimeError("This experiment requires an Open3D version supporting utility.random.seed.")
    plan["open3d_version"] = o3d.__version__
    plan["numpy_version"] = np.__version__
    plan["OMP_NUM_THREADS"] = 1
    plan["inputs"] = [{"method": m, "training_seed": s, "folder": str(v[2]), "snapshot_sha256": v[1].get("snapshot_sha256")} for (m, s), v in caches.items()]
    plan["entry_script_sha256"] = __import__("common").digest_file(__file__)
    write_json(args.output / "protocol.json", plan)
    keys = expected[:args.max_pairs] if args.max_pairs else expected
    paired_runs = []
    for seed in args.training_seeds:
        for budget in args.budgets:
            selected = {}
            for method in METHODS:
                selected[method] = []
                for key in keys:
                    with np.load(caches[(method, seed)][0][key], allow_pickle=False) as data:
                        gt = data["transform"].copy()
                        ref, src, inlier = correspondence_metrics(data["ref_corr_points"], data["src_corr_points"], gt, data["corr_scores"], budget)
                    if budget and len(ref) < budget:
                        raise ValueError(f"Requested {budget} correspondences but only {len(ref)} available: {key}")
                    selected[method].append((key, gt, ref, src, inlier))
            for b, m in zip(selected["baseline"], selected["mspki"]):
                if b[0] != m[0] or not np.allclose(b[1], m[1], atol=1e-5, rtol=0):
                    raise ValueError(f"Different pair identity or ground truth between methods: {b[0]}")
            for estimator_seed in args.estimator_seeds:
                by_method = {}
                for method in METHODS:
                    rows = []
                    started = time.perf_counter()
                    for pair_index, (key, gt, ref, src, inlier) in enumerate(selected[method], 1):
                        o3d.utility.random.seed(estimator_seed)
                        if len(src) < 4:
                            estimated = np.full((4, 4), np.nan)
                            valid = 0
                        else:
                            pc_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
                            pc_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref))
                            indices = np.arange(len(src), dtype=np.int32)
                            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                                pc_src, pc_ref, o3d.utility.Vector2iVector(np.column_stack([indices, indices])),
                                0.3, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
                                [], o3d.pipelines.registration.RANSACConvergenceCriteria(args.iterations, 0.999))
                            valid = int(len(result.correspondence_set) >= 4)
                            estimated = np.asarray(result.transformation) if valid else np.full((4, 4), np.nan)
                        rre, rte, tr = pose_metrics(gt, estimated)
                        rows.append({"seq": key[0], "src_frame": key[1], "ref_frame": key[2],
                                     "IR": float(inlier.mean()) if len(inlier) else 0.0,
                                     "correspondences": len(src), "inliers": int(inlier.sum()),
                                     "pose_valid": valid, "RRE": rre, "RTE_m": rte, "TR": tr})
                        if pair_index % 100 == 0:
                            print(f"{method} train={seed} top={budget or 'All'} ransac={estimator_seed}: {pair_index}/{len(keys)}", flush=True)
                    name = f"{method}_train{seed}_top{budget or 'All'}_ransac{estimator_seed}.csv"
                    write_csv(args.output / "pairs" / name, rows)
                    by_method[method] = rows
                    print(f"{name}: TR={100*np.mean([x['TR'] for x in rows]):.4f}%, {time.perf_counter()-started:.1f}s", flush=True)
                summary = paired_summary(by_method["baseline"], by_method["mspki"])
                summary.update(training_seed=seed, budget=budget, estimator_seed=estimator_seed)
                paired_runs.append(summary)
                write_csv(args.output / "paired_runs.csv", paired_runs)
    training_rows, aggregate_rows = [], []
    metric_names = [k for k in paired_runs[0] if k not in ("training_seed", "budget", "estimator_seed")]
    for budget in args.budgets:
        for seed in args.training_seeds:
            runs = [x for x in paired_runs if x["budget"] == budget and x["training_seed"] == seed]
            row = {"budget": budget, "training_seed": seed, "estimator_repeats": len(runs)}
            row.update({k: float(np.mean([x[k] for x in runs])) for k in metric_names})
            row["delta_IR_pp"] = row["mspki_IR_pct"] - row["baseline_IR_pct"]
            row["delta_TR_pp"] = row["mspki_TR_pct"] - row["baseline_TR_pct"]
            training_rows.append(row)
        group = [x for x in training_rows if x["budget"] == budget]
        aggregate = {"budget": budget, "training_seeds": len(group)}
        for metric in metric_names + ["delta_IR_pp", "delta_TR_pp"]:
            values = [x[metric] for x in group]
            aggregate[metric + "_mean"] = float(np.mean(values))
            aggregate[metric + "_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        aggregate_rows.append(aggregate)
    write_csv(args.output / "training_seed_means.csv", training_rows)
    write_csv(args.output / "summary.csv", aggregate_rows)
    plan["completed"] = True
    write_json(args.output / "protocol.json", plan)


if __name__ == "__main__":
    main()
