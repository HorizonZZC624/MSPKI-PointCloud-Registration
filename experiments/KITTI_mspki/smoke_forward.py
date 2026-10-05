
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pareconv.utils.torch import to_cuda

from config import make_cfg, make_experiment_parser
from dataset import test_data_loader
from model import create_model
from loss import Evaluator


def main():
    parser = make_experiment_parser(training=False, description="One-pair KITTI smoke forward.")
    args = parser.parse_args()
    cfg = make_cfg(args)

    loader, _ = test_data_loader(cfg)
    sample = next(iter(loader))
    sample = to_cuda(sample)

    model = create_model(cfg).cuda().eval()
    evaluator = Evaluator(cfg).cuda()
    with torch.inference_mode():
        output = model(sample, compute_gt=True, estimate_transform=True)
        metrics = evaluator(output, sample)

    print(f"variant={cfg.variant}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    for k, v in metrics.items():
        print(f"{k}={float(v):.6f}")
    print("SMOKE FORWARD: PASS")


if __name__ == "__main__":
    main()
