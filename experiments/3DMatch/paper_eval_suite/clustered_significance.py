
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "PIR": "coarse_precision",
    "IR": "inlier_ratio",
    "RR": "registration_accepted",
}


def make_parser():
    p = argparse.ArgumentParser(
        description="Scene-clustered paired significance analysis for Baseline vs MSPKI."
    )
    p.add_argument("--root", required=True,
                   help="Root containing baseline/ and mspki/ result trees.")
    p.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[7351, 2026, 3407])
    p.add_argument("--re_feature_source", default="official")
    p.add_argument("--bootstrap", type=int, default=100000)
    p.add_argument("--rng_seed", type=int, default=20260913)
    p.add_argument("--output_dir", required=True)
    return p


def csv_path(root, variant, re_source, seed, benchmark):
    return (Path(root) / variant / f"re_{re_source}" / f"seed_{seed}" /
            "registration" / benchmark / "pairs_fhp.csv")


def exact_scene_signflip(scene_values):
    x = np.asarray(scene_values, dtype=np.float64)
    observed = abs(float(x.mean()))
    if len(x) > 20:
        raise ValueError("Exact sign-flip is intended for <=20 scene clusters.")
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        stats.append(abs(float((x * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(stats) >= observed - 1e-15))


def main():
    args = make_parser().parse_args()
    keys = ["scene", "ref_frame", "src_frame"]
    per_seed = []

    for seed in args.seeds:
        bpath = csv_path(args.root, "baseline", args.re_feature_source, seed, args.benchmark)
        mpath = csv_path(args.root, "mspki", args.re_feature_source, seed, args.benchmark)
        if not bpath.is_file():
            raise FileNotFoundError(bpath)
        if not mpath.is_file():
            raise FileNotFoundError(mpath)

        b = pd.read_csv(bpath)
        m = pd.read_csv(mpath)

        required = keys + list(METRICS.values())
        for col in required:
            if col not in b.columns or col not in m.columns:
                raise KeyError(f"Missing required column: {col}")

        merged = b[required].merge(
            m[required], on=keys, how="inner", suffixes=("_baseline", "_mspki"),
            validate="one_to_one"
        )
        if len(merged) != len(b) or len(merged) != len(m):
            raise RuntimeError(
                f"Pair mismatch for seed {seed}: baseline={len(b)}, mspki={len(m)}, merged={len(merged)}"
            )

        for label, col in METRICS.items():
            merged[f"delta_{label}"] = merged[f"{col}_mspki"] - merged[f"{col}_baseline"]
        merged["seed"] = int(seed)
        per_seed.append(merged[keys + ["seed"] + [f"delta_{x}" for x in METRICS]])

    stacked = pd.concat(per_seed, ignore_index=True)



    pair_delta = (
        stacked.groupby(keys, as_index=False)[[f"delta_{x}" for x in METRICS]]
        .mean()
    )
    scene_delta = (
        pair_delta.groupby("scene", as_index=False)[[f"delta_{x}" for x in METRICS]]
        .mean()
    )

    rng = np.random.default_rng(args.rng_seed)
    rows = []
    for label in METRICS:
        vals = scene_delta[f"delta_{label}"].to_numpy(dtype=np.float64)
        observed = float(vals.mean())
        boot = rng.choice(vals, size=(int(args.bootstrap), len(vals)), replace=True).mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        p = exact_scene_signflip(vals)
        rows.append({
            "benchmark": args.benchmark,
            "metric": label,
            "num_scenes": int(len(vals)),
            "num_pairs": int(len(pair_delta)),
            "seeds": ",".join(map(str, args.seeds)),
            "macro_scene_delta": observed,
            "macro_scene_delta_pp": observed * 100.0,
            "cluster_bootstrap_ci95_low": float(low),
            "cluster_bootstrap_ci95_high": float(high),
            "cluster_bootstrap_ci95_low_pp": float(low * 100.0),
            "cluster_bootstrap_ci95_high_pp": float(high * 100.0),
            "exact_scene_signflip_p_two_sided": p,
            "positive_scenes": int((vals > 0).sum()),
            "negative_scenes": int((vals < 0).sum()),
            "zero_scenes": int((vals == 0).sum()),
        })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stacked.to_csv(out / "paired_seed_level_differences.csv", index=False)
    pair_delta.to_csv(out / "paired_pair_differences_seed_averaged.csv", index=False)
    scene_delta.to_csv(out / "scene_level_differences.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "clustered_significance_summary.csv", index=False)
    with (out / "clustered_significance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
