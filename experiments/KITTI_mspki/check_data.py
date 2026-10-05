
from __future__ import annotations

import argparse
import os.path as osp
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--metadata_root", required=True)
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    metadata_root = Path(args.metadata_root)
    expected_counts = {"train": 1358, "val": 180, "test": 555}
    ok = True

    for split, expected in expected_counts.items():
        meta_file = metadata_root / f"{split}.pkl"
        if not meta_file.is_file():
            print(f"[MISSING] {meta_file}")
            ok = False
            continue
        with meta_file.open("rb") as f:
            meta = pickle.load(f)
        print(f"[OK] {split}.pkl: {len(meta)} pairs (reference archive: {expected})")
        missing_clouds = 0
        for item in meta:
            if not (dataset_root / item["pcd0"]).is_file():
                missing_clouds += 1
            if not (dataset_root / item["pcd1"]).is_file():
                missing_clouds += 1
        print(f"     missing referenced point-cloud files: {missing_clouds}")
        if missing_clouds:
            ok = False

        if split == "train":
            missing_masks = 0
            for item in meta:
                mask = (
                    dataset_root
                    / "train_pair_overlap_masks"
                    / str(item["seq_id"])
                    / f"masks_{item['frame0']}_{item['frame1']}.npz"
                )
                if not mask.is_file():
                    missing_masks += 1
            print(f"     missing train overlap masks: {missing_masks}")
            if missing_masks:
                print("     -> run prepare_overlap_masks.py before official training.")

    print("\nDATA CHECK:", "PASS" if ok else "NEEDS PREPARATION")


if __name__ == "__main__":
    main()
