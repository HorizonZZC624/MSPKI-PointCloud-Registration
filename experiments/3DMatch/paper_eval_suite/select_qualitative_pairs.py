
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def make_parser():
    p = argparse.ArgumentParser(description="Select objective Baseline-vs-MSPKI qualitative candidates.")
    p.add_argument("--root", required=True)
    p.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], default="3DLoMatch")
    p.add_argument("--seed", type=int, default=7351)
    p.add_argument("--re_feature_source", default="official")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--low_overlap_threshold", type=float, default=0.20)
    p.add_argument("--output", required=True)
    return p


def path(root, variant, re_source, seed, benchmark):
    return (Path(root) / variant / f"re_{re_source}" / f"seed_{seed}" /
            "registration" / benchmark / "pairs_fhp.csv")


def top_rows(df, category, mask, sort_cols, ascending, topk):
    out = df.loc[mask].sort_values(sort_cols, ascending=ascending).head(topk).copy()
    out.insert(0, "category", category)
    return out


def main():
    args = make_parser().parse_args()
    keys = ["scene", "ref_frame", "src_frame"]
    cols = keys + ["overlap", "coarse_precision", "inlier_ratio", "registration_accepted"]

    b = pd.read_csv(path(args.root, "baseline", args.re_feature_source, args.seed, args.benchmark))
    m = pd.read_csv(path(args.root, "mspki", args.re_feature_source, args.seed, args.benchmark))

    x = b[cols].merge(m[cols], on=keys, suffixes=("_baseline", "_mspki"), validate="one_to_one")
    x["overlap"] = 0.5 * (x["overlap_baseline"] + x["overlap_mspki"])
    x["delta_ir"] = x["inlier_ratio_mspki"] - x["inlier_ratio_baseline"]
    x["delta_pir"] = x["coarse_precision_mspki"] - x["coarse_precision_baseline"]
    x["delta_rr"] = x["registration_accepted_mspki"] - x["registration_accepted_baseline"]

    keep = [
        "scene", "ref_frame", "src_frame", "overlap",
        "inlier_ratio_baseline", "inlier_ratio_mspki", "delta_ir",
        "coarse_precision_baseline", "coarse_precision_mspki", "delta_pir",
        "registration_accepted_baseline", "registration_accepted_mspki", "delta_rr",
    ]

    low = x["overlap"] < args.low_overlap_threshold
    rescue = low & (x["registration_accepted_baseline"] == 0) & (x["registration_accepted_mspki"] == 1)
    both_ok = low & (x["registration_accepted_baseline"] == 1) & (x["registration_accepted_mspki"] == 1)
    limitation = (x["delta_ir"] < 0) | (
        (x["registration_accepted_baseline"] == 1) & (x["registration_accepted_mspki"] == 0)
    )

    positive = x[x["delta_ir"] > 0].copy()
    median_gain = positive["delta_ir"].median() if len(positive) else 0.0
    x["distance_to_median_positive_gain"] = (x["delta_ir"] - median_gain).abs()

    groups = [
        top_rows(x, "low_overlap_rescue", rescue, ["delta_ir", "overlap"], [False, True], args.topk),
        top_rows(x, "low_overlap_both_success_gain", both_ok & (x["delta_ir"] > 0),
                 ["delta_ir", "overlap"], [False, True], args.topk),
        top_rows(x, "representative_positive_gain", x["delta_ir"] > 0,
                 ["distance_to_median_positive_gain", "overlap"], [True, True], args.topk),
        top_rows(x, "limitation_or_counterexample", limitation,
                 ["delta_ir", "overlap"], [True, True], args.topk),
    ]

    out = pd.concat([g[["category"] + keep] for g in groups if len(g)], ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
