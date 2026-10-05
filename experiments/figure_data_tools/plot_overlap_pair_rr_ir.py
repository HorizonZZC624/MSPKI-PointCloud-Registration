from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    groups: dict[str, list[dict[str, float]]] = {"baseline": [], "mspki": []}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            model = row["model"].strip().lower()
            if model not in groups:
                raise ValueError(f"Unexpected model in {path}: {model}")
            groups[model].append(
                {
                    key: float(row[key])
                    for key in (
                        "bin_low",
                        "bin_high",
                        "pairs",
                        "ir_mean_percent",
                        "ir_sample_sd_pp",
                        "rr_pair_mean_percent",
                        "rr_pair_sample_sd_pp",
                    )
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda row: row["bin_low"])
    if not groups["baseline"] or len(groups["baseline"]) != len(groups["mspki"]):
        raise ValueError("The CSV must contain matching Baseline and MSPKI overlap bins")
    for baseline, mspki in zip(groups["baseline"], groups["mspki"]):
        for key in ("bin_low", "bin_high", "pairs"):
            if baseline[key] != mspki[key]:
                raise ValueError(f"The two models do not share the same {key}")
    return groups


def draw(input_path: Path, output_path: Path, dpi: int) -> None:
    groups = read_rows(input_path)
    baseline = groups["baseline"]
    x = np.arange(len(baseline))
    labels = [f"{row['bin_low']:.2f}-{row['bin_high']:.2f}" for row in baseline]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.94))
    colors = {"baseline": "#555555", "mspki": "#1f77b4"}
    handles: dict[str, object] = {}
    for model, rows in groups.items():
        color = colors[model]
        for metric, mean_key, sd_key, linestyle, marker in (
            ("rr", "rr_pair_mean_percent", "rr_pair_sample_sd_pp", "-", "o"),
            ("ir", "ir_mean_percent", "ir_sample_sd_pp", "--", "s"),
        ):
            mean = np.array([row[mean_key] for row in rows])
            sd = np.array([row[sd_key] for row in rows])
            (line,) = ax.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.1,
                markersize=6.5,
            )
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
            handles[f"{model}_{metric}"] = line

    ax.set_xlim(-0.2, len(x) - 0.8)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Ground-truth overlap bin")
    ax.set_ylabel("Performance (%)")
    ax.grid(True, color="#cfcfcf", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    for position, row in zip(x, baseline):
        ax.text(position, 3.0, f"n={int(row['pairs'])}", ha="center", color="#666666", fontsize=9)

    ax.legend(
        [
            handles["baseline_rr"],
            handles["baseline_ir"],
            handles["mspki_rr"],
            handles["mspki_ir"],
        ],
        [
            r"Baseline $RR_{\mathrm{pair}}$",
            "Baseline IR",
            r"MSPKI $RR_{\mathrm{pair}}$",
            "MSPKI IR",
        ],
        loc="upper right",
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(pad=0.7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pair-level overlap IR and registration acceptance rate")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    draw(args.input, args.output, args.dpi)


if __name__ == "__main__":
    main()
