
from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pareconv.engine import SingleTester
from pareconv.engine.base_tester import inject_default_parser
from pareconv.utils.common import ensure_dir, get_log_string
from pareconv.utils.torch import release_cuda

from config import make_cfg, make_experiment_parser
from dataset import test_data_loader
from loss import Evaluator
from model import create_model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_parser():
    p = make_experiment_parser(training=False, description="KITTI official test extraction.")
    p.add_argument("--overwrite_features", action="store_true")
    p.add_argument("--density_keep_ratio", type=float, default=1.0)
    p.add_argument("--density_mask_seed", type=int, default=9017)
    return inject_default_parser(p)


class Tester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        loader, limits = test_data_loader(
            cfg,
            density_keep_ratio=self.args.density_keep_ratio,
            density_mask_seed=self.args.density_mask_seed,
        )
        self.register_loader(loader)
        self.logger.info(f"KNN limits: {limits}.")

        model = create_model(cfg).cuda()
        self.total_params = sum(p.numel() for p in model.parameters())
        self.register_model(model)
        self.evaluator = Evaluator(cfg).cuda()

        self.output_dir = cfg.feature_dir
        if osp.isdir(self.output_dir) and any(Path(self.output_dir).glob("*.npz")):
            if not self.args.overwrite_features:
                raise FileExistsError(
                    f"{self.output_dir} already contains NPZ files. Use --overwrite_features."
                )
            shutil.rmtree(self.output_dir)
        ensure_dir(self.output_dir)
        self.saved_pairs = 0
        self.expected_pairs = len(self.test_loader.dataset)
        self.manifest_file = osp.join(self.output_dir, "manifest.json")

    def before_test_epoch(self):
        self.manifest = {
            "code_version": self.cfg.code_version,
            "variant": self.cfg.variant,
            "seed": self.cfg.seed,
            "re_feature_source": self.cfg.fine_matching.re_feature_source,
            "fine_context": self.cfg.backbone.fine_context,
            "mspki_scales": list(self.cfg.backbone.mspki_neighbor_scales),
            "num_hypotheses": self.cfg.fine_matching.num_hypotheses,
            "dataset_split": "KITTI 08-10 test",
            "density_target": "source",
            "density_keep_ratio": self.args.density_keep_ratio,
            "density_mask_seed": self.args.density_mask_seed,
            "density_mask_rule": "SHA256(seed:sequence:source_frame), prefix of one permutation",
            "snapshot": osp.abspath(self.args.snapshot),
            "snapshot_sha256": sha256(self.args.snapshot),
            "parameters": int(self.total_params),
            "expected_pairs": int(self.expected_pairs),
            "saved_pairs": 0,
            "completed": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def test_step(self, iteration, data_dict):
        return self.model(data_dict, compute_gt=True, estimate_transform=True)

    def eval_step(self, iteration, data_dict, output_dict):
        return self.evaluator(output_dict, data_dict)

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return (
            f"seq={data_dict['seq_id']}, ref={data_dict['ref_frame']}, "
            f"src={data_dict['src_frame']}, " + get_log_string(result_dict=result_dict)
        )

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        fn = osp.join(
            self.output_dir,
            f"{int(data_dict['seq_id'])}_{int(data_dict['src_frame'])}_{int(data_dict['ref_frame'])}.npz",
        )
        np.savez_compressed(
            fn,
            ref_points=release_cuda(output_dict["ref_points"]),
            src_points=release_cuda(output_dict["src_points"]),
            ref_points_c=release_cuda(output_dict["ref_points_c"]),
            src_points_c=release_cuda(output_dict["src_points_c"]),
            ref_node_corr_indices=release_cuda(output_dict["ref_node_corr_indices"]),
            src_node_corr_indices=release_cuda(output_dict["src_node_corr_indices"]),
            ref_corr_points=release_cuda(output_dict["ref_corr_points"]),
            src_corr_points=release_cuda(output_dict["src_corr_points"]),
            corr_scores=release_cuda(output_dict["corr_scores"]),
            gt_node_corr_indices=release_cuda(output_dict["gt_node_corr_indices"]),
            estimated_transform=release_cuda(output_dict["estimated_transform"]),
            transform=release_cuda(data_dict["transform"]),
        )
        self.saved_pairs += 1

    def after_test_epoch(self):
        self.manifest["saved_pairs"] = int(self.saved_pairs)
        self.manifest["completed"] = self.saved_pairs == self.expected_pairs
        self.manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        if not self.manifest["completed"]:
            raise RuntimeError(
                f"Incomplete KITTI extraction: {self.saved_pairs}/{self.expected_pairs}"
            )


def main():
    parser = make_parser()
    known, _ = parser.parse_known_args()
    if not 0.0 < known.density_keep_ratio <= 1.0:
        parser.error("--density_keep_ratio must be in (0, 1].")
    if known.density_mask_seed < 0:
        parser.error("--density_mask_seed must be nonnegative.")
    if known.density_keep_ratio < 1.0 and not known.run_tag:
        parser.error("--run_tag is required for density stress runs.")
    cfg = make_cfg(known)
    Tester(cfg, parser).run()


if __name__ == "__main__":
    main()
