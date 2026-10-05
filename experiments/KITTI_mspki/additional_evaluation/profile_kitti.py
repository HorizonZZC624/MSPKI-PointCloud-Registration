from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from pareconv.engine import SingleTester
from pareconv.engine.base_tester import inject_default_parser

from config import make_cfg, make_experiment_parser
from dataset import test_data_loader
from model import create_model


class ProfileDone(Exception):
    pass


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description="KITTI inference-only latency and GPU-memory profiler."
    )
    parser.add_argument("--warmup_pairs", type=int, default=20)
    parser.add_argument("--profile_pairs", type=int, default=100)
    parser.add_argument("--profile_output", type=str, required=True)
    parser.add_argument("--density_keep_ratio", type=float, default=1.0)
    parser.add_argument("--density_mask_seed", type=int, default=9017)
    return inject_default_parser(parser)


class Profiler(SingleTester):
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
        self.total_params = int(sum(p.numel() for p in model.parameters()))
        self.register_model(model)

        self.warmup_pairs = int(self.args.warmup_pairs)
        self.profile_pairs = int(self.args.profile_pairs)
        self.total_pairs = self.warmup_pairs + self.profile_pairs

        self.seen_pairs = 0
        self.times_ms = []
        self.output_file = Path(self.args.profile_output)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self._written = False

    def before_test_epoch(self):
        self.model.eval()

                                                                   
                                                                            
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def test_step(self, iteration, data_dict):
                                                                                   
                                                                            
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output_dict = self.model(
                data_dict,
                compute_gt=False,
                estimate_transform=True,
            )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.seen_pairs += 1

        if self.seen_pairs > self.warmup_pairs:
            self.times_ms.append(float(elapsed_ms))

        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
                                                         
        return {}

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        phase = "warmup" if self.seen_pairs <= self.warmup_pairs else "timed"
        return (
            f"profile_pair={self.seen_pairs}/{self.total_pairs}, "
            f"phase={phase}"
        )

    def after_test_step(
        self,
        iteration,
        data_dict,
        output_dict,
        result_dict,
    ):
        if self.seen_pairs >= self.total_pairs:
            self.write_result()
            raise ProfileDone()

    def after_test_epoch(self):
        self.write_result()

    def write_result(self):
        if self._written:
            return

        torch.cuda.synchronize()

        if len(self.times_ms) != self.profile_pairs:
            raise RuntimeError(
                f"Expected {self.profile_pairs} timed pairs, "
                f"but obtained {len(self.times_ms)}."
            )

        arr = np.asarray(self.times_ms, dtype=np.float64)

        result = {
            "benchmark": "KITTI 08-10",
            "variant": str(self.cfg.variant),
            "seed": int(self.cfg.seed),
            "re_feature_source": str(
                self.cfg.fine_matching.re_feature_source
            ),
            "density_keep_ratio": float(self.args.density_keep_ratio),
            "density_mask_seed": int(self.args.density_mask_seed),
            "warmup_pairs": self.warmup_pairs,
            "profile_pairs": self.profile_pairs,
            "parameters": self.total_params,
            "latency_mean_ms": float(arr.mean()),
            "latency_sample_sd_ms_within_run": (
                float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            ),
            "latency_median_ms": float(np.median(arr)),
            "latency_min_ms": float(arr.min()),
            "latency_max_ms": float(arr.max()),
            "peak_gpu_memory_mib": float(
                torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            ),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.version() is not None
                else None
            ),
            "timing_scope": (
                "model forward + pose estimation; excludes data loading, "
                "host-to-device transfer, GT computation and result saving"
            ),
            "pair_times_ms": [float(x) for x in self.times_ms],
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("\n========== KITTI PROFILE RESULT ==========")
        print(f"Variant:     {result['variant']}")
        print(f"Parameters:  {result['parameters']:,}")
        print(
            f"Latency:     {result['latency_mean_ms']:.3f} ms/pair "
            f"(within-run SD {result['latency_sample_sd_ms_within_run']:.3f})"
        )
        print(
            f"Peak memory: {result['peak_gpu_memory_mib']:.2f} MiB"
        )
        print(f"GPU:         {result['gpu']}")
        print(f"Saved to:    {self.output_file}")
        print("==========================================\n")

        self._written = True


def main():
    parser = make_parser()
    known_args, _ = parser.parse_known_args()

    if known_args.warmup_pairs < 0:
        parser.error("--warmup_pairs must be >= 0.")
    if known_args.profile_pairs < 1:
        parser.error("--profile_pairs must be >= 1.")
    if not 0.0 < known_args.density_keep_ratio <= 1.0:
        parser.error("--density_keep_ratio must be in (0, 1].")

    cfg = make_cfg(known_args)

    try:
        Profiler(cfg, parser).run()
    except ProfileDone:
        pass


if __name__ == "__main__":
    main()
