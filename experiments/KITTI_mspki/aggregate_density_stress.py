

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


TRAINING_SEEDS = (2026, 3407, 7351)
VARIANTS = ("baseline", "mspki")
KEY_FIELDS = ("seq", "src_frame", "ref_frame")


def read_run(root: Path, variant: str, seed: int, ratio: float, mask_seed: int):
    tag = f"density_src_r{round(100 * ratio):03d}_m{mask_seed}"
    run_dir = root / variant / "re_official" / f"seed_{seed}" / tag
    manifest = json.loads((run_dir / "features" / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("completed") is not True
        or manifest.get("saved_pairs") != 555
        or manifest.get("expected_pairs") != 555
        or manifest.get("variant") != variant
        or manifest.get("seed") != seed
        or manifest.get("density_target") != "source"
        or abs(float(manifest.get("density_keep_ratio", -1)) - ratio) > 1e-9
        or manifest.get("density_mask_seed") != mask_seed
    ):
        raise ValueError(f"Incomplete or mismatched manifest: {run_dir}")

    path = run_dir / "registration" / "pairs_fhp.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 555:
        raise ValueError(f"Expected 555 pair rows, got {len(rows)}: {path}")
    keyed = {tuple(row[field] for field in KEY_FIELDS): row for row in rows}
    if len(keyed) != 555:
        raise ValueError(f"Duplicate pair keys: {path}")
    return keyed


def mean(values):
    return statistics.mean(values) if values else float("nan")


def compare(baseline, mspki, keys):
    b = [baseline[key] for key in keys]
    m = [mspki[key] for key in keys]
    b_ir = 100 * mean([float(row["IR"]) for row in b])
    m_ir = 100 * mean([float(row["IR"]) for row in m])
    b_tr = 100 * mean([float(row["TR"]) for row in b])
    m_tr = 100 * mean([float(row["TR"]) for row in m])
    common = [(br, mr) for br, mr in zip(b, m) if float(br["TR"]) > 0.5 and float(mr["TR"]) > 0.5]
    rescue = sum(float(br["TR"]) < 0.5 and float(mr["TR"]) > 0.5 for br, mr in zip(b, m))
    harm = sum(float(br["TR"]) > 0.5 and float(mr["TR"]) < 0.5 for br, mr in zip(b, m))
    return {
        "pairs": len(keys),
        "baseline_IR_pct": b_ir,
        "mspki_IR_pct": m_ir,
        "delta_IR_pp": m_ir - b_ir,
        "baseline_TR_pct": b_tr,
        "mspki_TR_pct": m_tr,
        "delta_TR_pp": m_tr - b_tr,
        "mspki_only_success": rescue,
        "baseline_only_success": harm,
        "common_success": len(common),
        "baseline_RRE_common_deg": mean([float(br["RRE"]) for br, _ in common]),
        "mspki_RRE_common_deg": mean([float(mr["RRE"]) for _, mr in common]),
        "delta_RRE_common_deg": mean([float(mr["RRE"]) - float(br["RRE"]) for br, mr in common]),
        "baseline_RTE_common_cm": 100 * mean([float(br["RTE_m"]) for br, _ in common]),
        "mspki_RTE_common_cm": 100 * mean([float(mr["RTE_m"]) for _, mr in common]),
        "delta_RTE_common_cm": 100 * mean([float(mr["RTE_m"]) - float(br["RTE_m"]) for br, mr in common]),
    }


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--keep_ratios", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--mask_seed", type=int, default=9017)
    args = parser.parse_args()

    per_seed = []
    per_sequence = []
    reference_keys = None
    for ratio in args.keep_ratios:
        for seed in TRAINING_SEEDS:
            b = read_run(args.output_root, "baseline", seed, ratio, args.mask_seed)
            m = read_run(args.output_root, "mspki", seed, ratio, args.mask_seed)
            if b.keys() != m.keys():
                raise ValueError(f"Baseline/MSPKI pair keys differ for ratio={ratio}, seed={seed}")
            if reference_keys is None:
                reference_keys = set(b)
            elif set(b) != reference_keys:
                raise ValueError(f"Pair keys differ across density runs for ratio={ratio}, seed={seed}")
            per_seed.append({"keep_ratio": ratio, "training_seed": seed, **compare(b, m, sorted(b))})
            for sequence in ("8", "9", "10"):
                keys = sorted(key for key in b if key[0] == sequence)
                per_sequence.append({
                    "keep_ratio": ratio,
                    "training_seed": seed,
                    "sequence": sequence,
                    **compare(b, m, keys),
                })

    metrics = (
        "baseline_IR_pct", "mspki_IR_pct", "delta_IR_pp",
        "baseline_TR_pct", "mspki_TR_pct", "delta_TR_pp",
        "delta_RRE_common_deg", "delta_RTE_common_cm",
    )
    summary = []
    for ratio in args.keep_ratios:
        runs = [row for row in per_seed if row["keep_ratio"] == ratio]
        result = {"keep_ratio": ratio, "training_seeds": len(runs), "pairs_per_seed": 555}
        for metric in metrics:
            values = [row[metric] for row in runs]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sample_sd"] = statistics.stdev(values)
        summary.append(result)

    target = args.output_root / f"density_analysis_m{args.mask_seed}"
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "per_training_seed.csv", per_seed)
    write_csv(target / "per_sequence.csv", per_sequence)
    write_csv(target / "three_seed_summary.csv", summary)
    print(f"Saved paired analysis for {len(summary)} densities and {len(reference_keys)} physical pairs: {target}")


if __name__ == "__main__":
    main()
