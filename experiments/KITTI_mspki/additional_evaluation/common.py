from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (2026, 3407, 7351)
METHODS = ("baseline", "mspki")


def metadata_keys(metadata_root):
    with (Path(metadata_root) / "test.pkl").open("rb") as stream:
        metadata = pickle.load(stream)
    keys = [(int(x["seq_id"]), int(x["frame1"]), int(x["frame0"])) for x in metadata]
    if len(keys) != 555 or len(set(keys)) != 555:
        raise ValueError("The KITTI test metadata must contain 555 unique pairs.")
    return metadata, sorted(keys)


def pair_key(path):
    return tuple(map(int, Path(path).stem.split("_")))


def cache_files(folder, expected):
    files = {pair_key(p): p for p in Path(folder).glob("*.npz")}
    if set(files) != set(expected):
        raise ValueError(f"Incomplete cache {folder}: {len(files)}/555 pairs; missing {len(set(expected) - set(files))}.")
    return files


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed_indices(count, ratio, mask_seed, sequence, source_frame):
    if not 0 < ratio <= 1 or count <= 0 or mask_seed < 0:
        raise ValueError("Invalid density mask settings.")
    if ratio == 1:
        return np.arange(count, dtype=np.int64)
    key = f"{mask_seed}:{sequence}:{source_frame}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.permutation(count)[:max(1, round(count * ratio))])


def correspondence_metrics(ref, src, gt, scores=None, budget=0):
    ref, src, gt = np.asarray(ref), np.asarray(src), np.asarray(gt)
    if ref.shape != src.shape or ref.ndim != 2 or ref.shape[1] != 3 or gt.shape != (4, 4):
        raise ValueError("Invalid correspondence or transform shapes.")
    if not all(np.isfinite(x).all() for x in (ref, src, gt)):
        raise ValueError("Non-finite coordinates or ground truth.")
    if budget > 0:
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(ref) or not np.isfinite(scores).all():
            raise ValueError("Invalid correspondence scores.")
        order = np.argsort(-scores, kind="stable")[:budget]
        ref, src = ref[order], src[order]
    transformed = src @ gt[:3, :3].T + gt[:3, 3]
    inlier = np.linalg.norm(ref - transformed, axis=1) < 1.0
    return ref, src, inlier


def pose_metrics(gt, estimated):
    gt, estimated = np.asarray(gt), np.asarray(estimated)
    if estimated.shape != (4, 4) or not np.isfinite(estimated).all():
        return float("nan"), float("nan"), 0
    cosine = (np.trace(estimated[:3, :3].T @ gt[:3, :3]) - 1) / 2
    rre = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    rte = float(np.linalg.norm(gt[:3, 3] - estimated[:3, 3]))
    return rre, rte, int(rre < 5.0 and rte < 2.0)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def paired_summary(baseline, proposed):
    b = {(int(x["seq"]), int(x["src_frame"]), int(x["ref_frame"])): x for x in baseline}
    m = {(int(x["seq"]), int(x["src_frame"]), int(x["ref_frame"])): x for x in proposed}
    if len(b) != len(baseline) or len(m) != len(proposed):
        raise ValueError("Duplicate pair records.")
    if not b or set(b) != set(m):
        raise ValueError("Pair sets differ between methods.")
    keys = sorted(b)
    common = [k for k in keys if int(float(b[k]["TR"])) == int(float(m[k]["TR"])) == 1]
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {
        "pairs": len(keys), "common_success_pairs": len(common),
        "baseline_IR_pct": 100 * mean([float(b[k]["IR"]) for k in keys]),
        "mspki_IR_pct": 100 * mean([float(m[k]["IR"]) for k in keys]),
        "baseline_TR_pct": 100 * mean([float(b[k]["TR"]) for k in keys]),
        "mspki_TR_pct": 100 * mean([float(m[k]["TR"]) for k in keys]),
        "recovered": sum(float(b[k]["TR"]) == 0 and float(m[k]["TR"]) == 1 for k in keys),
        "reversed": sum(float(b[k]["TR"]) == 1 and float(m[k]["TR"]) == 0 for k in keys),
        "baseline_common_RRE_deg": mean([float(b[k]["RRE"]) for k in common]),
        "mspki_common_RRE_deg": mean([float(m[k]["RRE"]) for k in common]),
        "baseline_common_RTE_cm": 100 * mean([float(b[k]["RTE_m"]) for k in common]),
        "mspki_common_RTE_cm": 100 * mean([float(m[k]["RTE_m"]) for k in common]),
    }
