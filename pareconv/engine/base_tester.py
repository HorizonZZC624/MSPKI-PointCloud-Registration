import abc
import argparse
import json
import os.path as osp
import sys
import time

import torch

from pareconv.engine.logger import Logger
from pareconv.utils.torch import initialize


def _torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def inject_default_parser(parser=None):
    pass
    if parser is None:
        parser = argparse.ArgumentParser()
    if '--snapshot' not in parser._option_string_actions:
        parser.add_argument('--snapshot', required=True, help='checkpoint path')
    return parser


class BaseTester(abc.ABC):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()
        self.cfg = cfg

        log_file = osp.join(
            cfg.log_dir,
            f'test-{time.strftime("%Y%m%d-%H%M%S")}.log',
        )
        self.logger = Logger(log_file=log_file)
        self.logger.info('Command executed: ' + ' '.join(sys.argv))
        self.logger.info('Configs:\n' + json.dumps(cfg, indent=4))

        self.args.snapshot = osp.abspath(osp.expanduser(self.args.snapshot))
        if not osp.isfile(self.args.snapshot):
            raise FileNotFoundError(self.args.snapshot)
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')

        self.cudnn_deterministic = bool(cudnn_deterministic)
        self.seed = int(cfg.seed)
        initialize(seed=self.seed, cudnn_deterministic=self.cudnn_deterministic)

        self.model = None
        self.iteration = None
        self.test_loader = None
        self.saved_states = {}
        self.checkpoint_metadata = {}

    def load_snapshot(self, snapshot):
        self.logger.info(f'Loading from "{snapshot}".')
        state_dict = _torch_load(snapshot, map_location='cpu')
        if 'model' not in state_dict:
            raise KeyError('Checkpoint does not contain a model state dictionary.')

        checkpoint_variant = state_dict.get('variant')
        if checkpoint_variant is not None and checkpoint_variant != self.cfg.variant:
            raise ValueError(
                f'Checkpoint variant={checkpoint_variant!r} does not match '
                f'--variant={self.cfg.variant!r}.'
            )
        checkpoint_re_source = state_dict.get('re_feature_source')
        current_re_source = self.cfg.fine_matching.re_feature_source
        if checkpoint_re_source is not None and checkpoint_re_source != current_re_source:
            raise ValueError(
                f'Checkpoint RE source={checkpoint_re_source!r} does not match '
                f'--re_feature_source={current_re_source!r}.'
            )
        checkpoint_seed = state_dict.get('seed')
        if checkpoint_seed is not None and int(checkpoint_seed) != int(self.cfg.seed):
            raise ValueError(
                f'Checkpoint seed={checkpoint_seed} does not match --seed={self.cfg.seed}. '
                'Use the checkpoint\'s actual seed so feature directories and statistics '
                'remain traceable.'
            )
        checkpoint_code_version = state_dict.get('code_version')
        runtime_code_version = getattr(self.cfg, 'code_version', None)
        if (
            checkpoint_code_version is not None
            and runtime_code_version is not None
            and checkpoint_code_version != runtime_code_version
        ):
            raise ValueError(
                f'Checkpoint code_version={checkpoint_code_version!r} does not match '
                f'runtime code_version={runtime_code_version!r}. Re-test a checkpoint '
                'trained with the final code or explicitly migrate its metadata only '
                'after verifying state compatibility.'
            )

        self.model.load_state_dict(state_dict['model'], strict=True)
        self.saved_states.update(state_dict.get('saved_states', {}))
        self.checkpoint_metadata = state_dict
        self.logger.info('Model has been loaded.')

    def register_model(self, model):
        self.model = model
        self.logger.info('Model description:\n' + str(model))
        return model

    def register_loader(self, test_loader):
        self.test_loader = test_loader

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError
