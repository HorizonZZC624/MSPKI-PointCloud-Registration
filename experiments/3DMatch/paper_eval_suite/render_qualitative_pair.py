
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_parser():
    p = argparse.ArgumentParser(description="Render Baseline-vs-MSPKI registration/correspondence qualitative panels.")
    p.add_argument("--root", required=True)
    p.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], default="3DLoMatch")
    p.add_argument("--seed", type=int, default=7351)
    p.add_argument("--re_feature_source", default="official")
    p.add_argument("--scene", required=True)
    p.add_argument("--ref_frame", type=int, required=True)
    p.add_argument("--src_frame", type=int, required=True)
    p.add_argument("--max_points", type=int, default=5000)
    p.add_argument("--max_corr", type=int, default=150)
    p.add_argument("--inlier_radius", type=float, default=0.10)
    p.add_argument("--output", required=True)
    return p


def npz_path(root, variant, re_source, seed, benchmark, scene, ref_frame, src_frame):
    return (Path(root) / variant / f"re_{re_source}" / f"seed_{seed}" /
            "features" / benchmark / scene / f"{ref_frame}_{src_frame}.npz")


def transform_points(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def sample_points(points, n):
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, num=n, dtype=np.int64)
    return points[idx]


def equal_axes(ax, xyz):
    if len(xyz) == 0:
        return
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float((hi - lo).max()) * 0.55, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def choose_corr(d, max_corr):
    ref = np.asarray(d["ref_corr_points"])
    src = np.asarray(d["src_corr_points"])
    scores = np.asarray(d["corr_scores"]).reshape(-1) if "corr_scores" in d.files else np.ones(len(ref))
    if len(ref) > max_corr:
        idx = np.argsort(scores)[-max_corr:]
        ref, src, scores = ref[idx], src[idx], scores[idx]
    return ref, src, scores


def panel_alignment(ax, d, title, max_points):
    ref = sample_points(np.asarray(d["ref_points"]), max_points)
    src = sample_points(np.asarray(d["src_points"]), max_points)
    est = np.asarray(d["estimated_transform"])
    src_aligned = transform_points(src, est)
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=0.6, alpha=0.45)
    ax.scatter(src_aligned[:, 0], src_aligned[:, 1], src_aligned[:, 2], s=0.6, alpha=0.45)
    ax.set_title(title)
    equal_axes(ax, np.concatenate([ref, src_aligned], axis=0))


def panel_corr(ax, d, title, max_points, max_corr, inlier_radius):
    refp = sample_points(np.asarray(d["ref_points"]), max_points)
    srcp = sample_points(np.asarray(d["src_points"]), max_points)
    gt = np.asarray(d["transform"])
    srcp_gt = transform_points(srcp, gt)
    ax.scatter(refp[:, 0], refp[:, 1], refp[:, 2], s=0.4, alpha=0.20)
    ax.scatter(srcp_gt[:, 0], srcp_gt[:, 1], srcp_gt[:, 2], s=0.4, alpha=0.20)

    ref, src, _ = choose_corr(d, max_corr)
    src_gt = transform_points(src, gt)
    distances = np.linalg.norm(ref - src_gt, axis=1)
    inlier = distances < inlier_radius

    for a, b, ok in zip(ref, src_gt, inlier):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                linewidth=0.5 if ok else 0.25, alpha=0.55 if ok else 0.25)

    ir = float(inlier.mean()) if len(inlier) else 0.0
    ax.set_title(f"{title} | shown IR={100*ir:.1f}%")
    equal_axes(ax, np.concatenate([refp, srcp_gt], axis=0))


def main():
    args = make_parser().parse_args()
    b = load(npz_path(args.root, "baseline", args.re_feature_source, args.seed, args.benchmark,
                      args.scene, args.ref_frame, args.src_frame))
    m = load(npz_path(args.root, "mspki", args.re_feature_source, args.seed, args.benchmark,
                      args.scene, args.ref_frame, args.src_frame))

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    panel_alignment(ax1, b, "Baseline registration", args.max_points)
    panel_alignment(ax2, m, "MSPKI registration", args.max_points)
    panel_corr(ax3, b, "Baseline correspondences", args.max_points, args.max_corr, args.inlier_radius)
    panel_corr(ax4, m, "MSPKI correspondences", args.max_points, args.max_corr, args.inlier_radius)

    overlap = float(np.asarray(m["overlap"]).reshape(-1)[0]) if "overlap" in m.files else float("nan")
    fig.suptitle(
        f"{args.benchmark} | {args.scene} | {args.ref_frame}-{args.src_frame} | overlap={overlap:.3f}"
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
