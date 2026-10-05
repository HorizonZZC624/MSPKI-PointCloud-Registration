from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from common import ROOT, SEEDS, METHODS, cache_files, digest_file, metadata_keys


def main():
    p = argparse.ArgumentParser(description="Re-extract missing KITTI density coordinates with trained checkpoints, without overwriting paper results.")
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/KITTI")
    p.add_argument("--metadata-root", type=Path, default=ROOT / "data/KITTI/metadata")
    p.add_argument("--checkpoint-root", type=Path, default=ROOT / "output/KITTI_mspki")
    p.add_argument("--output-root", type=Path, default=ROOT / "output/KITTI_additional/density_cache/KITTI_mspki")
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, .75, .5, .25])
    p.add_argument("--training-seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--mask-seed", type=int, default=9017)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output_root.resolve() == args.checkpoint_root.resolve():
        raise ValueError("Use a separate output root to preserve the original paper runs.")
    if any(not 0 < x <= 1 for x in args.ratios):
        raise ValueError("Density ratios must be in (0, 1].")
    metadata, expected = metadata_keys(args.metadata_root)
    missing = {str(args.dataset_root / item[key]) for item in metadata for key in ("pcd0", "pcd1") if not (args.dataset_root / item[key]).is_file()}
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test scans, e.g. {sorted(missing)[0]}")
    tasks, archive_members = [], []
    for ratio in args.ratios:
        tag = f"density_src_r{round(100*ratio):03d}_m{args.mask_seed}"
        for method in METHODS:
            for seed in args.training_seeds:
                checkpoint = args.checkpoint_root / method / "re_official" / f"seed_{seed}/snapshots/best.pth.tar"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                run_dir = args.output_root / method / "re_official" / f"seed_{seed}" / tag
                folder = run_dir / "features"
                complete = False
                if (folder / "manifest.json").is_file():
                    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                    try:
                        cache_files(folder, expected)
                        complete = (manifest.get("completed") is True and manifest.get("saved_pairs") == 555
                                    and manifest.get("variant") == method and manifest.get("seed") == seed
                                    and manifest.get("snapshot_sha256") == digest_file(checkpoint)
                                    and manifest.get("density_keep_ratio") == ratio and manifest.get("density_mask_seed") == args.mask_seed)
                    except ValueError:
                        complete = False
                if not complete and list(folder.glob("*.npz")):
                    raise ValueError(f"Partial or mismatched cache: {folder}. Choose a new output root; no automatic deletion.")
                common = ["--variant", method, "--seed", str(seed), "--re_feature_source", "official",
                          "--dataset_root", str(args.dataset_root), "--metadata_root", str(args.metadata_root),
                          "--output_root", str(args.output_root), "--run_tag", tag, "--num_workers", str(args.num_workers)]
                tasks.append((complete,
                              [sys.executable, str(ROOT / "experiments/KITTI_mspki/test.py"), *common,
                               "--snapshot", str(checkpoint), "--density_keep_ratio", str(ratio), "--density_mask_seed", str(args.mask_seed)],
                              [sys.executable, str(ROOT / "experiments/KITTI_mspki/eval.py"), *common]))
                archive_members.extend([run_dir / "features/manifest.json", run_dir / "registration/pairs_fhp.csv", run_dir / "registration/summary_fhp.json"])
    for complete, extraction, evaluation in tasks:
        print(json.dumps({"extraction_required": not complete, "test_command": extraction, "eval_command": evaluation}), flush=True)
        if not args.dry_run:
            if not complete:
                subprocess.run(extraction, cwd=ROOT, check=True)
            subprocess.run(evaluation, cwd=ROOT, check=True)
    if not args.dry_run:
        destination = args.output_root.parent / "density_analysis_complete.tar.gz"
        temporary = destination.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            for path in archive_members:
                tar.add(path, arcname="output/KITTI_mspki/" + path.relative_to(args.output_root).as_posix())
        temporary.replace(destination)
        print(f"New matched CSV/manifest archive: {destination}", flush=True)


if __name__ == "__main__":
    main()
