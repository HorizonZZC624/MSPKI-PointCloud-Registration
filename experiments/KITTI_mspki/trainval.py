
from __future__ import annotations

import json
import math
import os.path as osp
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim

from pareconv.engine import EpochBasedTrainer
from pareconv.engine.base_trainer import inject_default_parser
from pareconv.utils.common import print_model_parameters

from config import make_cfg, make_experiment_parser
from dataset import train_valid_data_loader
from loss import Evaluator, OverallLoss
from model import create_model


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _existing_files(path):
    if not osp.isdir(path):
        return []
    return [str(p) for p in Path(path).rglob("*") if p.is_file()]


def _prepare(cfg, args):
    existing = _existing_files(cfg.output_dir)
    if args.overwrite_run:
        shutil.rmtree(cfg.output_dir, ignore_errors=True)
        for p in (cfg.output_dir, cfg.snapshot_dir, cfg.log_dir, cfg.event_dir, cfg.feature_dir, cfg.registration_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
        return
    if existing and not args.resume:
        raise FileExistsError(
            f"Run directory already contains files: {cfg.output_dir}\n"
            "Use --resume, --overwrite_run, or another --run_tag."
        )


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, parser):
        super().__init__(
            cfg,
            max_epoch=cfg.optim.max_epoch,
            parser=parser,
            run_grad_check=False,
            autograd_anomaly_detection=False,
            grad_acc_steps=cfg.optim.grad_acc_steps,
        )
        self.best_rr = None
        self.best_selection_key = None

        start = time.time()
        train_loader, val_loader, limits = train_valid_data_loader(cfg, self.distributed)
        self.logger.info(f"Data loaders created in {time.time()-start:.3f}s.")
        self.logger.info(f"KNN limits: {limits}.")
        self.register_loader(train_loader, val_loader)

        model = self.register_model(create_model(cfg).cuda())
        print_model_parameters(model)
        total = sum(p.numel() for p in model.parameters())
        self.logger.critical(
            f"KITTI official split | variant={cfg.variant}, seed={cfg.seed}, "
            f"params={total:,}, fine_context={cfg.backbone.fine_context}, "
            f"RE={cfg.fine_matching.re_feature_source}, voxel={cfg.backbone.init_voxel_size}"
        )

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
        self.register_optimizer(optimizer)
        self.register_scheduler(
            optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.optim.lr_decay_steps,
                gamma=cfg.optim.lr_decay,
            )
        )

        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

        if self.local_rank == 0:
            with open(osp.join(cfg.output_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
                json.dump(_json_safe(cfg), f, indent=2)

    def train_step(self, epoch, iteration, data_dict):
        output = self.model(
            data_dict,
            compute_gt=True,
            estimate_transform=self.cfg.model.estimate_transform_during_training,
        )
        result = self.loss_func(output, data_dict)
        result.update(self.evaluator(output, data_dict, evaluate_fine=False, evaluate_registration=False))
        return output, result

    def val_step(self, epoch, iteration, data_dict):
        output = self.model(data_dict, compute_gt=True, estimate_transform=True)
        return output, self.evaluator(output, data_dict)

    def after_val_epoch(self, epoch):
        if self.local_rank != 0:
            return

        rr = float(self.val_summary["RR"])
        rre = float(self.val_summary.get("RRE", float("inf")))
        rte = float(self.val_summary.get("RTE", float("inf")))
        ir = float(self.val_summary.get("IR", float("-inf")))




        rre_key = -rre if math.isfinite(rre) else float("-inf")
        rte_key = -rte if math.isfinite(rte) else float("-inf")
        ir_key = ir if math.isfinite(ir) else float("-inf")
        selection_key = (rr, rre_key, rte_key, ir_key)

        restored_key = self.saved_states.get("best_selection_key")
        if restored_key is not None:
            self.best_selection_key = tuple(float(x) for x in restored_key)
            self.best_rr = float(self.best_selection_key[0])

        if self.best_selection_key is None or selection_key > self.best_selection_key:
            self.best_selection_key = selection_key
            self.best_rr = rr
            self.saved_states["best_rr"] = rr
            self.saved_states["best_selection_key"] = list(selection_key)
            self.saved_states["best_epoch"] = int(epoch)

            checkpoint = self.build_checkpoint(include_optimizer=False)
            checkpoint["best_rr"] = rr
            checkpoint["best_selection_key"] = list(selection_key)
            checkpoint["best_epoch"] = int(epoch)
            destination = osp.join(self.snapshot_dir, "best.pth.tar")
            torch.save(checkpoint, destination)

            with open(osp.join(self.snapshot_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "code_version": self.cfg.code_version,
                        "variant": self.cfg.variant,
                        "seed": self.cfg.seed,
                        "best_epoch": int(epoch),
                        "selection_rule": [
                            "maximize validation TR",
                            "tie: minimize validation RRE",
                            "tie: minimize validation RTE",
                            "tie: maximize validation IR"
                        ],
                        "selection_key": list(selection_key),
                        "validation": _json_safe(self.val_summary),
                        "checkpoint": destination,
                    },
                    f,
                    indent=2,
                )
            self.logger.critical(
                f"New KITTI best: epoch={epoch}, val TR={rr:.6f}, "
                f"RRE={rre:.6f}, RTE={rte:.6f}, IR={ir:.6f}"
            )
        else:
            self.saved_states["best_rr"] = float(self.best_rr)
            self.saved_states["best_selection_key"] = list(self.best_selection_key)


def main():
    parser = inject_default_parser(
        make_experiment_parser(training=True, description="Train KITTI Baseline/MSPKI under official PARE-Net settings.")
    )
    args = parser.parse_args()
    cfg = make_cfg(args)
    _prepare(cfg, args)
    Trainer(cfg, parser).run()


if __name__ == "__main__":
    main()
