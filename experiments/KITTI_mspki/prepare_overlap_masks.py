
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pareconv.utils.registration import compute_overlap_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--metadata_root", required=True)
    p.add_argument("--matching_radius", type=float, default=0.6)
    args = p.parse_args()

    with open(osp.join(args.metadata_root, "train.pkl"), "rb") as f:
        metadata = pickle.load(f)

    created = 0
    skipped = 0
    for item in tqdm(metadata, desc="KITTI train overlap masks"):
        seq = str(item["seq_id"])
        ref = item["frame0"]
        src = item["frame1"]
        out_dir = Path(args.dataset_root) / "train_pair_overlap_masks" / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"masks_{ref}_{src}.npz"
        if out.is_file():
            skipped += 1
            continue

        ref_points = np.load(Path(args.dataset_root) / item["pcd0"])
        src_points = np.load(Path(args.dataset_root) / item["pcd1"])
        ref_mask, src_mask = compute_overlap_mask(
            ref_points,
            src_points,
            item["transform"],
            positive_radius=args.matching_radius,
        )
        np.savez_compressed(out, ref_masks=ref_mask, src_masks=src_mask)
        created += 1

    print(f"created={created}, skipped_existing={skipped}, total={len(metadata)}")


if __name__ == "__main__":
    main()
