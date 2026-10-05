from __future__ import annotations

import json
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

from config import ensure_run_dirs, make_cfg, make_experiment_parser
from dataset import train_valid_data_loader
from loss import Evaluator, OverallLoss
from model import create_model


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _existing_run_files(output_dir):
    if not osp.isdir(output_dir):
        return []
    return [str(path) for path in Path(output_dir).rglob('*') if path.is_file()]


def _prepare_run_directory(cfg, args):
    if args.resume and args.overwrite_run:
        raise ValueError('--resume and --overwrite_run are mutually exclusive.')
    existing = _existing_run_files(cfg.output_dir)
    if args.overwrite_run:
        shutil.rmtree(cfg.output_dir, ignore_errors=True)
        ensure_run_dirs(cfg)
        return
    if existing and not args.resume:
        preview = '\n'.join(existing[:5])
        raise FileExistsError(
            'Run directory already contains files:\n'
            f'{cfg.output_dir}\nExamples:\n{preview}\n'
            'Use --resume, --overwrite_run, another seed, or another --run_tag.'
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

        start = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(
            cfg, self.distributed
        )
        self.logger.info(f'Data loaders created in {time.time() - start:.3f}s.')
        self.logger.info(f'KNN limits: {neighbor_limits}.')
        self.register_loader(train_loader, val_loader)

        model = create_model(cfg).cuda()
        model = self.register_model(model)
        print_model_parameters(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.critical(
            f'Variant={cfg.variant} (canonical={cfg.canonical_variant}), seed={cfg.seed}, '
            f'params={total_params:,}, trainable={trainable_params:,}, '
            f'coarse_context={cfg.backbone.coarse_context}, '
            f'fine_context={cfg.backbone.fine_context}, '
            f'RE-source={cfg.fine_matching.re_feature_source}, '
            f'SPSA-ratio={cfg.backbone.spsa_partial_ratio}, '
            f'MSPKI-scales={cfg.backbone.mspki_neighbor_scales}, '
            f'MSPKI-fusion={cfg.backbone.mspki_fusion}, '
            f'LayerScale={cfg.backbone.layer_scale_init}'
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
            with open(osp.join(cfg.output_dir, 'resolved_config.json'), 'w', encoding='utf-8') as f:
                json.dump(_json_safe(cfg), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _check_finite(result_dict, epoch, iteration, stage):
        invalid = {
            k: v for k, v in result_dict.items()
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
        }
        if invalid:
            raise FloatingPointError(
                f'Non-finite {stage} values at epoch={epoch}, iteration={iteration}: {invalid}'
            )

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(
            data_dict,
            compute_gt=True,
            estimate_transform=self.cfg.model.estimate_transform_during_training,
        )
        result_dict = self.loss_func(output_dict, data_dict)
        result_dict.update(
            self.evaluator(
                output_dict,
                data_dict,
                evaluate_fine=False,
                evaluate_registration=False,
            )
        )
        self._check_finite(result_dict, epoch, iteration, 'training')
        return output_dict, result_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict, compute_gt=True, estimate_transform=True)






        result_dict = self.evaluator(output_dict, data_dict)
        self._check_finite(result_dict, epoch, iteration, 'validation')
        return output_dict, result_dict

    def after_val_epoch(self, epoch):
        if self.local_rank != 0:
            return
        rr = float(self.val_summary['RR'])
        restored = self.saved_states.get('best_rr')
        if restored is not None:
            self.best_rr = float(restored)
        if self.best_rr is None or rr > self.best_rr:
            self.best_rr = rr
            self.saved_states['best_rr'] = rr
            self.saved_states['best_epoch'] = int(epoch)

            destination = osp.join(self.snapshot_dir, 'best.pth.tar')
            checkpoint = self.build_checkpoint(include_optimizer=False)
            checkpoint['best_rr'] = rr
            checkpoint['best_epoch'] = int(epoch)
            torch.save(checkpoint, destination)

            metrics = {
                'code_version': self.cfg.code_version,
                'variant': self.cfg.variant,
                'canonical_variant': self.cfg.canonical_variant,
                'seed': self.cfg.seed,
                're_feature_source': self.cfg.fine_matching.re_feature_source,
                'mspki_scales': list(self.cfg.backbone.mspki_neighbor_scales),
                'mspki_fusion': self.cfg.backbone.mspki_fusion,
                'best_epoch': int(epoch),
                'validation': _json_safe(self.val_summary),
                'checkpoint': destination,
            }
            with open(osp.join(self.snapshot_dir, 'best_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            self.logger.critical(f'New best checkpoint: epoch={epoch}, RR={rr:.6f}')
        else:
            self.saved_states['best_rr'] = float(self.best_rr)


def main():
    parser = make_experiment_parser(
        training=True,
        description='Train an indoor PARE-Net or MSPKI variant.',
    )
    parser = inject_default_parser(parser)
    args = parser.parse_args()
    cfg = make_cfg(args)
    _prepare_run_directory(cfg, args)
    Trainer(cfg, parser).run()


if __name__ == '__main__':
    main()
