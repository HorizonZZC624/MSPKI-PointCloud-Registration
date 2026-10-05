from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from common import ROOT, SEEDS, METHODS, cache_files, correspondence_metrics, digest_file, metadata_keys, paired_summary, pose_metrics, write_csv, write_json


def read_archive(archive, ratios, seeds, mask_seed, expected):
    runs = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = {x.name: x for x in tar.getmembers() if x.isfile()}
        for ratio in ratios:
            tag = f"density_src_r{round(100*ratio):03d}_m{mask_seed}"
            for seed in seeds:
                for method in METHODS:
                    suffix = f"KITTI_mspki/{method}/re_official/seed_{seed}/{tag}/registration/pairs_fhp.csv"
                    matched = [name for name in members if name.endswith(suffix)]
                    if len(matched) != 1:
                        raise ValueError(f"Expected one archive member ending in {suffix}.")
                    name = matched[0]
                    manifest_name = name.rsplit("/registration/", 1)[0] + "/features/manifest.json"
                    with tar.extractfile(members[manifest_name]) as stream:
                        manifest = json.load(stream)
                    if (not manifest.get("completed") or manifest.get("saved_pairs") != 555
                            or manifest.get("variant") != method or manifest.get("seed") != seed
                            or abs(float(manifest.get("density_keep_ratio", -1))-ratio) > 1e-9
                            or manifest.get("density_mask_seed") != mask_seed):
                        raise ValueError(f"Mismatched archive manifest: {manifest_name}")
                    with tar.extractfile(members[name]) as stream:
                        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
                    keyed = {(int(x["seq"]), int(x["src_frame"]), int(x["ref_frame"])): x for x in rows}
                    if len(rows) != 555 or set(keyed) != set(expected):
                        raise ValueError(f"Incomplete or duplicated pair records: {name}")
                    runs[(ratio, seed, method)] = (keyed, manifest)
    return runs


def spatial_metrics(points, inlier, voxel_size):
    points = np.asarray(points, dtype=np.float64)
    accepted = points[inlier]
    all_cells = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0)
    cells = np.unique(np.floor(accepted / voxel_size).astype(np.int64), axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(accepted, rowvar=False))[::-1] if len(accepted) >= 3 else np.zeros(3)
    eigenvalues = np.maximum(eigenvalues, 0)
    return {"inlier_count": int(inlier.sum()), "correspondence_count": len(points),
            "inlier_occupied_cells": len(cells),
            "inlier_cell_fraction": len(cells) / len(all_cells) if len(all_cells) else 0.0,
            "inlier_spread_rms_m": float(np.sqrt(eigenvalues.sum())),
            "inlier_lambda2_lambda1": float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0,
            "inlier_lambda3_lambda1": float(eigenvalues[2] / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0}


def main():
    p = argparse.ArgumentParser(description="Paired KITTI density transitions and optional inlier geometry.")
    p.add_argument("--archive", type=Path, default=ROOT / "results/additional_evaluation/kitti_density_complete.tar.gz")
    p.add_argument("--cache-root", type=Path, default=ROOT / "output/KITTI_mspki")
    p.add_argument("--metadata-root", type=Path, default=ROOT / "data/KITTI/metadata")
    p.add_argument("--output", type=Path, default=ROOT / "output/KITTI_additional/density_diagnostics")
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, .75, .5, .25])
    p.add_argument("--training-seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--mask-seed", type=int, default=9017)
    p.add_argument("--geometry", action="store_true", help="Requires matching NPZ caches; never infers geometry from CSV.")
    p.add_argument("--voxel-size", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-pairs", type=int, default=0)
    args = p.parse_args()
    if args.voxel_size <= 0 or args.max_pairs < 0 or any(not 0 < x <= 1 for x in args.ratios):
        raise ValueError("Invalid diagnostic parameters.")
    _, expected = metadata_keys(args.metadata_root)
    runs = read_archive(args.archive, args.ratios, args.training_seeds, args.mask_seed, expected)
    geometry_files, missing = {}, []
    for ratio, seed, method in runs:
        tag = f"density_src_r{round(100*ratio):03d}_m{args.mask_seed}"
        folder = args.cache_root / method / "re_official" / f"seed_{seed}" / tag / "features"
        try:
            geometry_files[(ratio, seed, method)] = cache_files(folder, expected)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            archived = runs[(ratio, seed, method)][1]
            for field in ("variant", "seed", "snapshot_sha256", "density_keep_ratio", "density_mask_seed", "re_feature_source"):
                if manifest.get(field) != archived.get(field):
                    raise ValueError(f"Cache provenance differs for {field}: {folder}")
        except (ValueError, FileNotFoundError) as error:
            missing.append({"folder": str(folder), "reason": str(error)})
    plan = {"archive": str(args.archive), "archive_sha256": digest_file(args.archive),
            "ratios": args.ratios, "training_seeds": args.training_seeds,
            "complete_test_set": args.max_pairs == 0, "geometry_requested": args.geometry,
            "geometry_voxel_size_m": args.voxel_size, "missing_geometry_caches": missing,
            "geometry_definition": "Inlier cell fraction is occupied inlier cells divided by occupied cells of all predicted correspondences in the reference frame, not whole-scene coverage. Covariance ratios are descriptive; planar structure alone does not imply pose failure."}
    print(json.dumps({"archive": str(args.archive), "ratios": args.ratios, "training_seeds": args.training_seeds,
                      "geometry_requested": args.geometry, "missing_geometry_caches": len(missing),
                      "example_missing": missing[:1]}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.geometry and missing:
        raise ValueError("Missing or mismatched density NPZ caches. Restore original caches first; CSV alone supports transitions, not spatial analysis.")
    keys = expected[:args.max_pairs] if args.max_pairs else expected
    pairs, summaries, spatial_rows = [], [], []
    for ratio in args.ratios:
        for seed in args.training_seeds:
            b, m = runs[(ratio, seed, "baseline")][0], runs[(ratio, seed, "mspki")][0]
            summary = paired_summary([b[k] for k in keys], [m[k] for k in keys])
            summary.update(keep_ratio=ratio, training_seed=seed)
            summary["delta_IR_pp"] = summary["mspki_IR_pct"] - summary["baseline_IR_pct"]
            summary["delta_TR_pp"] = summary["mspki_TR_pct"] - summary["baseline_TR_pct"]
            summaries.append(summary)
            for key in keys:
                bs, ms = int(float(b[key]["TR"])), int(float(m[key]["TR"]))
                transition = {(1, 1): "both_success", (0, 1): "recovered", (1, 0): "reversed", (0, 0): "both_failure"}[(bs, ms)]
                row = {"keep_ratio": ratio, "training_seed": seed, "seq": key[0], "src_frame": key[1], "ref_frame": key[2],
                       "transition": transition, "baseline_IR": float(b[key]["IR"]), "mspki_IR": float(m[key]["IR"]),
                       "delta_IR_pp": 100*(float(m[key]["IR"])-float(b[key]["IR"])),
                       "baseline_TR": bs, "mspki_TR": ms,
                       "baseline_RRE": float(b[key]["RRE"]), "mspki_RRE": float(m[key]["RRE"]),
                       "baseline_RTE_m": float(b[key]["RTE_m"]), "mspki_RTE_m": float(m[key]["RTE_m"])}
                pairs.append(row)
                if args.geometry:
                    for method in METHODS:
                        cached = geometry_files[(ratio, seed, method)][key]
                        archived = runs[(ratio, seed, method)][0][key]
                        with np.load(cached, allow_pickle=False) as data:
                            ref, src, inlier = correspondence_metrics(data["ref_corr_points"], data["src_corr_points"], data["transform"])
                            rre, rte, tr = pose_metrics(data["transform"], data["estimated_transform"])
                        if abs(float(inlier.mean()) - float(archived["IR"])) > 1e-6 or tr != int(float(archived["TR"])) or abs(rte - float(archived["RTE_m"])) > 1e-4:
                            raise ValueError(f"NPZ metrics do not match the archived density run: {cached}. Keep newly inferred runs separate from archived results.")
                        detail = {"keep_ratio": ratio, "training_seed": seed, "method": method,
                                  "seq": key[0], "src_frame": key[1], "ref_frame": key[2], "transition": transition,
                                  "IR": float(inlier.mean()), "TR": tr}
                        detail.update(spatial_metrics(ref, inlier, args.voxel_size))
                        spatial_rows.append(detail)
    write_csv(args.output / "pair_transitions.csv", pairs)
    write_csv(args.output / "paired_training_seed_summary.csv", summaries)
    aggregates = []
    for ratio in args.ratios:
        group = [x for x in summaries if x["keep_ratio"] == ratio]
        row = {"keep_ratio": ratio, "training_seeds": len(group),
               "recovered_pair_seed_events": sum(x["recovered"] for x in group),
               "reversed_pair_seed_events": sum(x["reversed"] for x in group)}
        for metric in ("baseline_IR_pct", "mspki_IR_pct", "delta_IR_pp", "baseline_TR_pct", "mspki_TR_pct", "delta_TR_pp",
                       "baseline_common_RRE_deg", "mspki_common_RRE_deg", "baseline_common_RTE_cm", "mspki_common_RTE_cm"):
            values = [x[metric] for x in group]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        aggregates.append(row)
    write_csv(args.output / "summary.csv", aggregates)
    if .5 in args.ratios and .25 in args.ratios:
        changes = []
        for seed in args.training_seeds:
            for method in METHODS:
                half = runs[(.5, seed, method)][0]
                quarter = runs[(.25, seed, method)][0]
                for key in keys:
                    hs, qs = int(float(half[key]["TR"])), int(float(quarter[key]["TR"]))
                    changes.append({"training_seed": seed, "method": method, "seq": key[0], "src_frame": key[1], "ref_frame": key[2],
                                    "TR_r050": hs, "TR_r025": qs, "success_at50_failure_at25": int(hs == 1 and qs == 0),
                                    "IR_r050": float(half[key]["IR"]), "IR_r025": float(quarter[key]["IR"]),
                                    "RRE_r050": float(half[key]["RRE"]), "RRE_r025": float(quarter[key]["RRE"]),
                                    "RTE_m_r050": float(half[key]["RTE_m"]), "RTE_m_r025": float(quarter[key]["RTE_m"])})
        write_csv(args.output / "same_pair_density50_to25.csv", changes)
    if spatial_rows:
        write_csv(args.output / "inlier_geometry_pairs.csv", spatial_rows)
        groups = []
        fields = list(spatial_metrics(np.zeros((3, 3)), np.ones(3, dtype=bool), args.voxel_size))
        for ratio in args.ratios:
            for method in METHODS:
                for transition in ("both_success", "recovered", "reversed", "both_failure"):
                    selected = [x for x in spatial_rows if x["keep_ratio"] == ratio and x["method"] == method and x["transition"] == transition]
                    if selected:
                        group = {"keep_ratio": ratio, "method": method, "transition": transition, "pair_seed_events": len(selected)}
                        group.update({field + "_mean": float(np.mean([x[field] for x in selected])) for field in fields})
                        groups.append(group)
        write_csv(args.output / "geometry_transition_groups.csv", groups)
    plan["completed"] = True
    write_json(args.output / "protocol.json", plan)
    print(f"Saved diagnostics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
