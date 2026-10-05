
from __future__ import annotations

import argparse
import csv
import glob
import json
import os.path as osp
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pareconv.engine import Logger
from pareconv.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)

from config import make_cfg, make_experiment_parser


def make_parser():
    p = make_experiment_parser(training=False, description="Evaluate KITTI official test split.")
    p.add_argument("--method", default="fhp", choices=["fhp"])
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg(args)

    Path(cfg.registration_dir).mkdir(parents=True, exist_ok=True)
    log_file = osp.join(cfg.log_dir, f"eval-{time.strftime('%Y%m%d-%H%M%S')}.log")
    logger = Logger(log_file=log_file)

    files = sorted(
        glob.glob(osp.join(cfg.feature_dir, "*.npz")),
        key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split("_")],
    )
    if not files:
        raise FileNotFoundError(f"No KITTI npz files under {cfg.feature_dir}")

    rows = []
    for fn in files:
        seq, src_frame, ref_frame = [
            int(x) for x in osp.splitext(osp.basename(fn))[0].split("_")
        ]
        d = np.load(fn)
        coarse = evaluate_sparse_correspondences(
            d["ref_points_c"],
            d["src_points_c"],
            d["ref_node_corr_indices"],
            d["src_node_corr_indices"],
            d["gt_node_corr_indices"],
        )
        fine = evaluate_correspondences(
            d["ref_corr_points"],
            d["src_corr_points"],
            d["transform"],
            positive_radius=cfg.eval.acceptance_radius,
        )
        rre, rte = compute_registration_error(
            d["transform"], d["estimated_transform"]
        )
        accepted = bool(
            rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
        )
        rows.append(
            {
                "seq": seq,
                "src_frame": src_frame,
                "ref_frame": ref_frame,
                "PIR": float(coarse["precision"]),
                "IR": float(fine["inlier_ratio"]),
                "FMR": float(fine["inlier_ratio"] >= cfg.eval.inlier_ratio_threshold),
                "RRE": float(rre),
                "RTE_m": float(rte),
                "TR": float(accepted),
            }
        )

    pir = np.mean([r["PIR"] for r in rows])
    ir = np.mean([r["IR"] for r in rows])
    fmr = np.mean([r["FMR"] for r in rows])
    tr = np.mean([r["TR"] for r in rows])
    accepted = [r for r in rows if r["TR"] > 0.5]
    mean_rre = float(np.mean([r["RRE"] for r in accepted])) if accepted else float("nan")
    mean_rte = float(np.mean([r["RTE_m"] for r in accepted])) if accepted else float("nan")

    summary = {
        "protocol": "PARE-Net KITTI official split; TR if RRE<5deg and RTE<2m",
        "variant": cfg.variant,
        "seed": cfg.seed,
        "num_pairs": len(rows),
        "PIR": float(pir),
        "FMR": float(fmr),
        "IR": float(ir),
        "TR": float(tr),
        "RRE_deg_successful_pairs": mean_rre,
        "RTE_m_successful_pairs": mean_rte,
        "RTE_cm_successful_pairs": mean_rte * 100.0,
    }

    logger.critical(
        "KITTI | "
        f"PIR={100*pir:.3f}%, FMR={100*fmr:.3f}%, IR={100*ir:.3f}%, "
        f"TR={100*tr:.3f}%, RRE={mean_rre:.4f}deg, RTE={100*mean_rte:.3f}cm"
    )

    out_csv = osp.join(cfg.registration_dir, "pairs_fhp.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(osp.join(cfg.registration_dir, "summary_fhp.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
