import abc
import argparse
import glob
import json
import os
import os.path as osp
import random
import socket
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from pareconv.engine.logger import Logger
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.torch import all_reduce_tensors, initialize, release_cuda

try:
    import wandb
except ImportError:
    wandb = None


def _torch_load(path, map_location='cpu'):
    pass
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def inject_default_parser(parser=None):
    pass
    if parser is None:
        parser = argparse.ArgumentParser()

    existing = parser._option_string_actions
    arguments = (
        (('--resume',), dict(action='store_true', help='resume optimizer/scheduler state')),
        (('--debug',), dict(action='store_true', help='disable Weights & Biases logging')),
        (('--snapshot',), dict(default=None, help='checkpoint or training snapshot path')),
        (('--epoch',), dict(type=int, default=None, help='reserved compatibility argument')),
        (('--log_steps',), dict(type=int, default=30, help='logging interval')),
        (
            ('--local_rank', '--local-rank'),
            dict(
                dest='local_rank',
                type=int,
                default=int(os.environ.get('LOCAL_RANK', -1)),
                help='local rank for DDP',
            ),
        ),
        (('--model_desc',), dict(type=str, default='default_configs')),
    )
    for options, kwargs in arguments:
        if not any(option in existing for option in options):
            parser.add_argument(*options, **kwargs)
    return parser


def _plain_config(value):
    if isinstance(value, dict):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_commit(root_dir):
    try:
        return subprocess.check_output(
            ['git', '-C', root_dir, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        cfg,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()
        self.cfg = cfg

        if getattr(self.args, 'overwrite_run', False) and self.args.resume:
            raise ValueError('--overwrite_run and --resume are mutually exclusive.')

        time_stamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = osp.join(cfg.log_dir, f'train-{time_stamp}.log')
        self.logger = Logger(log_file=log_file, local_rank=self.args.local_rank)
        self.logger.info('Command executed: ' + ' '.join(sys.argv))
        self.logger.info('Configs:\n' + json.dumps(_plain_config(cfg), indent=4))

        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.distributed = self.args.local_rank != -1
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.local_rank = self.args.local_rank
            self.logger.info(
                f'Using DistributedDataParallel mode (world_size={self.world_size}).'
            )
        else:
            if torch.cuda.device_count() > 1:
                self.logger.warning(
                    'Multiple GPUs detected. Use torchrun/DistributedDataParallel '
                    'instead of DataParallel.'
                )
            self.world_size = 1
            self.local_rank = 0
            self.logger.info('Using Single-GPU mode.')

        self.cudnn_deterministic = bool(cudnn_deterministic)
        self.autograd_anomaly_detection = bool(autograd_anomaly_detection)
        self.seed = int(cfg.seed) + self.local_rank
        initialize(
            seed=self.seed,
            cudnn_deterministic=self.cudnn_deterministic,
            autograd_anomaly_detection=self.autograd_anomaly_detection,
        )

        self.resume_snapshot = self._resolve_resume_snapshot()


        self.snapshot_dir = (
            osp.dirname(osp.abspath(self.resume_snapshot))
            if self.resume_snapshot is not None
            else cfg.snapshot_dir
        )
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.log_steps = int(self.args.log_steps)
        self.run_grad_check = bool(run_grad_check)
        self.save_all_snapshots = bool(save_all_snapshots)
        self.grad_acc_steps = int(grad_acc_steps)
        if self.grad_acc_steps < 1:
            raise ValueError('grad_acc_steps must be >= 1.')

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.iteration = 0
        self.inner_iteration = 0
        self.train_loader = None
        self.val_loader = None
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)
        self.timer = Timer()
        self.saved_states = {}
        self.training = True

        self.debug = bool(self.args.debug)
        if wandb is None and not self.debug:
            self.debug = True
            self.logger.warning(
                'wandb is not installed; continuing with local logs only.'
            )
        if not self.debug and self.local_rank == 0:
            wandb.init(project='PARE-Net', name=f'{self.args.model_desc}_{time_stamp}')
            self.logger.info(f'Wandb is enabled. Event directory: {cfg.event_dir}.')

    def _resolve_resume_snapshot(self):
        if not self.args.resume:
            return None
        if self.args.snapshot is not None:
            snapshot = osp.abspath(osp.expanduser(self.args.snapshot))
            if not osp.isfile(snapshot):
                raise FileNotFoundError(snapshot)
            expected_snapshot_dir = osp.realpath(self.cfg.snapshot_dir)
            actual_snapshot_dir = osp.realpath(osp.dirname(snapshot))
            if actual_snapshot_dir != expected_snapshot_dir:
                raise ValueError(
                    'A resumable snapshot must belong to the current protocol run '
                    'directory so logs, best metrics, and checkpoints cannot split '
                    'across output roots. Expected parent '
                    f'{expected_snapshot_dir!r}, got {actual_snapshot_dir!r}. '
                    'Use the matching --output_root/variant/seed/RE source.'
                )
            return snapshot

        candidates = glob.glob(
            osp.join(self.cfg.snapshot_dir, '**', 'snapshot.pth.tar'),
            recursive=True,
        )
        if not candidates:
            raise FileNotFoundError(
                '--resume was requested, but no snapshot.pth.tar was found. '
                'Pass --snapshot explicitly.'
            )
        return max(candidates, key=osp.getmtime)

    def _model_state_dict(self):
        if self.model is None:
            raise RuntimeError('Model has not been registered.')
        model_state_dict = self.model.state_dict()
        if self.distributed:
            model_state_dict = OrderedDict(
                (key[7:], value) for key, value in model_state_dict.items()
            )
        return model_state_dict

    def _rng_state(self):
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            return
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if 'cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])

    def _checkpoint_metadata(self):
        gpu_name = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return {
            'code_version': getattr(self.cfg, 'code_version', None),
            'variant': getattr(self.cfg, 'variant', None),
            'seed': getattr(self.cfg, 'seed', None),
            're_feature_source': getattr(
                getattr(self.cfg, 'fine_matching', {}),
                're_feature_source',
                None,
            ),
            'config': _plain_config(self.cfg),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'gpu_name': gpu_name,
            'hostname': socket.gethostname(),
            'git_commit': _git_commit(getattr(self.cfg, 'root_dir', '.')),
        }

    def build_checkpoint(self, include_optimizer=False):
        if self.optimizer is None:
            raise RuntimeError('Optimizer has not been registered.')
        state_dict = {
            'epoch': int(self.epoch),
            'iteration': int(self.iteration),
            'model': self._model_state_dict(),
            'saved_states': dict(self.saved_states),
            'rng_state': self._rng_state(),
            **self._checkpoint_metadata(),
        }
        if include_optimizer:
            state_dict['optimizer'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def save_snapshot(self, filename):
        if self.local_rank != 0:
            return
        if self.model is None or self.optimizer is None:
            raise RuntimeError('Model and optimizer must be registered before saving.')

        model_filename = osp.join(self.snapshot_dir, filename)
        torch.save(self.build_checkpoint(include_optimizer=False), model_filename)
        self.logger.info(f'Model saved to "{model_filename}".')

        snapshot_filename = osp.join(self.snapshot_dir, 'snapshot.pth.tar')
        torch.save(self.build_checkpoint(include_optimizer=True), snapshot_filename)
        self.logger.info(f'Training snapshot saved to "{snapshot_filename}".')

    def load_snapshot(self, snapshot, fix_prefix=True):
        snapshot = osp.abspath(osp.expanduser(snapshot))
        if not osp.isfile(snapshot):
            raise FileNotFoundError(snapshot)
        self.logger.info(f'Loading from "{snapshot}".')
        state_dict = _torch_load(snapshot, map_location='cpu')
        if 'model' not in state_dict:
            raise KeyError('Checkpoint does not contain the "model" key.')

        checkpoint_variant = state_dict.get('variant')
        if checkpoint_variant is not None and checkpoint_variant != self.cfg.variant:
            raise ValueError(
                f'Checkpoint variant={checkpoint_variant!r} does not match '
                f'current variant={self.cfg.variant!r}.'
            )
        checkpoint_re_source = state_dict.get('re_feature_source')
        current_re_source = self.cfg.fine_matching.re_feature_source
        if checkpoint_re_source is not None and checkpoint_re_source != current_re_source:
            raise ValueError(
                f'Checkpoint RE source={checkpoint_re_source!r} does not match '
                f'current RE source={current_re_source!r}.'
            )
        checkpoint_seed = state_dict.get('seed')
        if checkpoint_seed is not None and int(checkpoint_seed) != int(self.cfg.seed):
            message = (
                f'Checkpoint seed={checkpoint_seed} does not match current seed={self.cfg.seed}.'
            )
            if self.args.resume:
                raise ValueError(message)
            self.logger.warning(message + ' Loading as initialization only.')
        checkpoint_code_version = state_dict.get('code_version')
        runtime_code_version = getattr(self.cfg, 'code_version', None)
        if (
            checkpoint_code_version is not None
            and runtime_code_version is not None
            and checkpoint_code_version != runtime_code_version
        ):
            message = (
                f'Checkpoint code_version={checkpoint_code_version!r} does not match '
                f'current code_version={runtime_code_version!r}.'
            )
            if self.args.resume:
                raise ValueError(message)
            self.logger.warning(message + ' Loading weights as initialization only.')

        model_dict = state_dict['model']
        if fix_prefix and self.distributed:
            model_dict = OrderedDict(('module.' + key, value) for key, value in model_dict.items())
        self.model.load_state_dict(model_dict, strict=True)
        self.logger.info('Model has been loaded.')

        if self.args.resume:
            self.epoch = int(state_dict.get('epoch', self.epoch))
            self.iteration = int(state_dict.get('iteration', self.iteration))
            if 'optimizer' not in state_dict:
                raise KeyError(
                    'Resume requires optimizer state. Use snapshot.pth.tar rather '
                    'than epoch-*.pth.tar or best.pth.tar.'
                )
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.logger.info('Optimizer has been loaded.')
            if self.scheduler is not None:
                if 'scheduler' not in state_dict:
                    raise KeyError('Resume requires scheduler state.')
                self.scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info('Scheduler has been loaded.')
            self.saved_states.update(state_dict.get('saved_states', {}))
            self._restore_rng_state(state_dict.get('rng_state'))
            self.logger.info('Auxiliary training and RNG states have been loaded.')
        else:


            self.epoch = 0
            self.iteration = 0
            self.saved_states.clear()

        self.logger.info(f'Restored epoch={self.epoch}, iteration={self.iteration}.')

    def register_model(self, model):
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        self.model = model
        self.logger.info('Model description:\n' + str(model))
        return model

    def register_optimizer(self, optimizer):
        if self.distributed:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.world_size
        self.optimizer = optimizer

    def register_scheduler(self, scheduler):
        self.scheduler = scheduler

    def register_loader(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save_state(self, key, value):
        self.saved_states[key] = release_cuda(value)

    def read_state(self, key):
        return self.saved_states[key]

    def check_invalid_gradients(self):
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                self.logger.error(f'Non-finite gradient in parameter: {name}')
                return False
        return True

    def release_tensors(self, result_dict):
        if self.distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        return release_cuda(result_dict)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def write_event(self, phase, event_dict, index):
        if self.local_rank != 0 or self.debug:
            return
        wandb.log(
            {f'{phase}/{key}': value for key, value in event_dict.items()},
            step=index,
        )

    def finish_tracking(self):
        if not self.debug and self.local_rank == 0:
            wandb.finish()

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError
