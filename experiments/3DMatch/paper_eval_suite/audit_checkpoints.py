from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import torch


VARIANTS = (
    "baseline", "spsa", "mspki", "full", "spsa_fine",
    "mspki_coarse", "swapped", "capacity_matched"
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manuscript PARE-Net best checkpoints.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--re_feature_source", default="official")
    args = parser.parse_args()

    root = Path(args.checkpoint_root).expanduser().resolve()
    rows: List[Dict] = []
    errors: List[str] = []
    model_key_sets = {}

    for variant in args.variants:
        path = (
            root
            / variant
            / f"re_{args.re_feature_source}"
            / f"seed_{args.seed}"
            / "snapshots"
            / "best.pth.tar"
        )
        if not path.is_file():
            errors.append(f"Missing checkpoint: {path}")
            continue
        checkpoint = _load(path)
        if not isinstance(checkpoint, dict):
            errors.append(f"Checkpoint is not a dictionary: {path}")
            continue
        model = checkpoint.get("model")
        if not isinstance(model, dict):
            errors.append(f"Checkpoint has no model state dict: {path}")
            continue

        metadata_variant = checkpoint.get("variant")
        metadata_seed = checkpoint.get("seed")
        metadata_re = checkpoint.get("re_feature_source")
        if metadata_variant is not None and metadata_variant != variant:
            errors.append(
                f"Variant mismatch for {path}: metadata={metadata_variant}, expected={variant}"
            )
        if metadata_seed is not None and int(metadata_seed) != int(args.seed):
            errors.append(
                f"Seed mismatch for {path}: metadata={metadata_seed}, expected={args.seed}"
            )
        if metadata_re is not None and metadata_re != args.re_feature_source:
            errors.append(
                f"RE-source mismatch for {path}: metadata={metadata_re}, "
                f"expected={args.re_feature_source}"
            )

        model_keys = sorted(model.keys())
        model_key_sets[variant] = set(model_keys)
        num_tensors = sum(1 for value in model.values() if torch.is_tensor(value))
        num_state_values = sum(
            int(value.numel()) for value in model.values() if torch.is_tensor(value)
        )
        rows.append(
            {
                "variant": variant,
                "path": str(path),
                "sha256": _sha256(path),
                "file_size_bytes": int(path.stat().st_size),
                "checkpoint_variant": metadata_variant,
                "checkpoint_seed": metadata_seed,
                "checkpoint_re_feature_source": metadata_re,
                "code_version": checkpoint.get("code_version"),
                "best_epoch": checkpoint.get("best_epoch"),
                "best_rr": checkpoint.get("best_rr"),
                "epoch": checkpoint.get("epoch"),
                "num_model_keys": len(model_keys),
                "num_model_tensors": num_tensors,
                "num_model_state_values": num_state_values,
            }
        )

    baseline_keys = model_key_sets.get("baseline", set())
    key_differences = {}
    for variant, keys in model_key_sets.items():
        key_differences[variant] = {
            "extra_vs_baseline": sorted(keys - baseline_keys),
            "missing_vs_baseline": sorted(baseline_keys - keys),
        }

    payload = {
        "checkpoint_root": str(root),
        "seed": args.seed,
        "re_feature_source": args.re_feature_source,
        "checkpoints": rows,
        "state_key_differences": key_differences,
        "errors": errors,
        "passed": not errors and len(rows) == len(args.variants),
    }
    output_dir = Path(args.results_root).expanduser().resolve() / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "checkpoint_audit.json"
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Checkpoint audit saved to: {output_file}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
