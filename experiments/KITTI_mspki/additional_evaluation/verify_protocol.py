from __future__ import annotations

import ast
import csv
import hashlib
import json
import py_compile
from pathlib import Path

import numpy as np

from common import ROOT, correspondence_metrics, fixed_indices, pose_metrics
from density_diagnostics import spatial_metrics


def main():
    for path in Path(__file__).parent.glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    source = ast.parse((ROOT / "experiments/KITTI_mspki/dataset.py").read_text(encoding="utf-8"))
    definition = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == "FixedSourceDensityDataset")
    namespace = {"np": np, "hashlib": hashlib, "Dataset": object}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "existing_density_dataset", "exec"), namespace)
    original = namespace["FixedSourceDensityDataset"]
    for sequence in ("08", 8, "10"):
        for count in (3, 101, 30001):
            sample = {"seq_id": sequence, "src_frame": 29, "src_points": np.column_stack([np.arange(count)]*3), "src_feats": np.ones((count, 1))}
            for ratio in (.25, .5, .75, 1.0):
                expected = original([sample], ratio, 9017)[0]["src_points"]
                indices = fixed_indices(count, ratio, 9017, sequence, 29)
                np.testing.assert_array_equal(sample["src_points"][indices], expected)
    gt = np.eye(4)
    ref = np.array([[0., 0, 0], [1., 0, 0], [.999, 0, 0]])
    _, _, inlier = correspondence_metrics(ref, np.zeros((3, 3)), gt)
    np.testing.assert_array_equal(inlier, [True, False, True])
    moved = gt.copy()
    moved[0, 3] = 2.0
    assert pose_metrics(gt, moved)[2] == 0
    assert pose_metrics(gt, gt)[2] == 1
    assert pose_metrics(gt, np.full((4, 4), np.nan))[2] == 0
    line = np.column_stack([np.arange(20), np.zeros((20, 2))])
    geometry = spatial_metrics(line, np.ones(20, dtype=bool), 2.)
    assert geometry["inlier_count"] == 20 and geometry["inlier_lambda2_lambda1"] == 0
    summary_path = ROOT / "results/additional_evaluation/density_geometry/summary.csv"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as stream:
            summary = {float(x["keep_ratio"]): x for x in csv.DictReader(stream)}
        assert int(summary[.5]["recovered_pair_seed_events"]) == 13
        assert int(summary[.5]["reversed_pair_seed_events"]) == 3
        assert int(summary[.25]["recovered_pair_seed_events"]) == 30
        assert int(summary[.25]["reversed_pair_seed_events"]) == 39
        assert abs(float(summary[.5]["delta_TR_pp_mean"]) - 1000/1665) < 1e-10
        assert abs(float(summary[.25]["delta_TR_pp_mean"]) + 900/1665) < 1e-10
    print(json.dumps({"syntax": "passed", "existing_density_mask_equivalence": "passed", "metric_boundaries": "passed",
                      "known_density_transitions": "passed" if summary_path.is_file() else "not_checked",
                      "GeoTransformer_GPU_inference": "not_run_by_this_check; see archived complete-test protocol for reported inference"}, indent=2))


if __name__ == "__main__":
    main()
