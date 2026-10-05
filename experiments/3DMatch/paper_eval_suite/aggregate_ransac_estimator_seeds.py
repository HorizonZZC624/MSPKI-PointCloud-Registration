
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"summary_(ransac(?:_n(?P<n>\d+))?)_s(?P<seed>-?\d+)\.json$")

def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_registration_dir", required=True)
    p.add_argument("--mspki_registration_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p

def load_dir(root):
    root = Path(root)
    out = {}
    for f in root.glob("summary_ransac*_s*.json"):
        m = PATTERN.match(f.name)
        if not m:
            continue
        budget = "all" if m.group("n") is None else int(m.group("n"))
        seed = int(m.group("seed"))
        d = json.loads(f.read_text(encoding="utf-8"))
        out[(budget, seed)] = d
    if not out:
        raise FileNotFoundError(f"No seeded RANSAC summaries under {root}")
    return out

def mean_sd(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0

def main():
    args = parser().parse_args()
    b = load_dir(args.baseline_registration_dir)
    m = load_dir(args.mspki_registration_dir)

    if set(b) != set(m):
        raise RuntimeError(
            "Baseline/MSPKI seeded summary sets differ.\n"
            f"Only baseline: {sorted(set(b)-set(m))}\n"
            f"Only MSPKI: {sorted(set(m)-set(b))}"
        )

    budgets = sorted({k[0] for k in b}, key=lambda x: (x == "all", 10**9 if x == "all" else x))
    rows = []
    detailed = {}

    for budget in budgets:
        seeds = sorted(seed for (bud, seed) in b if bud == budget)
        per_seed = []
        for seed in seeds:
            bo = b[(budget, seed)]["overall"]
            mo = m[(budget, seed)]["overall"]
            row = {
                "budget": budget,
                "estimator_seed": seed,
                "baseline_RR_pct": 100.0 * float(bo["RR"]),
                "mspki_RR_pct": 100.0 * float(mo["RR"]),
                "delta_RR_pp": 100.0 * (float(mo["RR"]) - float(bo["RR"])),
                "baseline_IR_pct": 100.0 * float(bo["IR"]),
                "mspki_IR_pct": 100.0 * float(mo["IR"]),
                "baseline_FMR_pct": 100.0 * float(bo["FMR"]),
                "mspki_FMR_pct": 100.0 * float(mo["FMR"]),
            }
            rows.append(row)
            per_seed.append(row)

        brr = [r["baseline_RR_pct"] for r in per_seed]
        mrr = [r["mspki_RR_pct"] for r in per_seed]
        drr = [r["delta_RR_pp"] for r in per_seed]
        b_mean, b_sd = mean_sd(brr)
        m_mean, m_sd = mean_sd(mrr)
        d_mean, d_sd = mean_sd(drr)
        detailed[str(budget)] = {
            "seeds": seeds,
            "baseline_RR_mean_pct": b_mean,
            "baseline_RR_sd_pct": b_sd,
            "mspki_RR_mean_pct": m_mean,
            "mspki_RR_sd_pct": m_sd,
            "delta_RR_mean_pp": d_mean,
            "delta_RR_sd_pp": d_sd,
            "positive_delta_seeds": int(sum(x > 0 for x in drr)),
            "nonnegative_delta_seeds": int(sum(x >= 0 for x in drr)),
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "ransac_seeded_per_run.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    (out / "ransac_seeded_summary.json").write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(detailed, indent=2, ensure_ascii=False))
    print("Saved:", out)

if __name__ == "__main__":
    main()
