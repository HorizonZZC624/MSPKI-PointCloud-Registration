














from __future__ import annotations

import argparse
import math
import os
import os.path as osp
from typing import Optional, Sequence, Tuple

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

from pareconv.utils.common import ensure_dir


CODE_VERSION = 'mspki-official-protocol-v1'


VARIANTS = (
    'baseline',
    'spsa',
    'mspki',
    'pki',
    'full',
    'intended',
    'spsa_fine',
    'mspki_coarse',
    'pki_coarse',
    'swapped',
    'capacity_matched',
)

ROTATION_MODES = ('none', 'so3')
RE_FEATURE_SOURCES = ('official', 'decoder', 'encoder')






def add_experiment_args(parser: argparse.ArgumentParser, training: bool = False):
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--dataset_root', type=str, default=os.environ.get('PARENET_3DMATCH_ROOT'))
    parser.add_argument('--metadata_root', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--run_tag', type=str, default='')



    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--train_workers', type=int, default=None)
    parser.add_argument('--test_workers', type=int, default=None)

    parser.add_argument(
        '--re_feature_source',
        choices=RE_FEATURE_SOURCES,
        default='official',
        help=(
            "official = PARE-Net behavior (decoder RE in train mode, encoder RE "
            "in eval mode); decoder = decoder RE for train/val/test; encoder = "
            "encoder RE for train/val/test. Reported experiments use official."
        ),
    )


    parser.add_argument('--spsa_ratio', type=float, default=0.50)
    parser.add_argument('--spsa_qk_channels', type=int, default=32)
    parser.add_argument('--spsa_ffn_ratio', type=float, default=2.0)
    parser.add_argument('--mspki_scales', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--mspki_branch_channels', type=int, default=16)
    parser.add_argument('--mspki_ffn_ratio', type=float, default=2.0)
    parser.add_argument('--layer_scale_init', type=float, default=0.10)


    parser.add_argument('--rotation_mode', choices=ROTATION_MODES, default='none')
    parser.add_argument('--rotation_seed', type=int, default=7351)
    parser.add_argument('--num_hypotheses', type=int, default=None)
    parser.add_argument('--knn_query_chunk_size', type=int, default=None)
    parser.add_argument('--knn_support_chunk_size', type=int, default=None)

    if training:
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--max_epoch', type=int, default=40)
        parser.add_argument(
            '--skip_train_fhp',
            action='store_true',
            help=(
                'Skip final FHP pose estimation during train iterations.  FHP is '
                'inside no_grad and is not part of the loss, so this is an '
                'engineering speed option; leave it disabled for the strictest '
                'official-protocol audit.'
            ),
        )
        parser.add_argument('--overwrite_run', action='store_true')
    return parser


def make_experiment_parser(training: bool = False, description: Optional[str] = None):
    return add_experiment_args(argparse.ArgumentParser(description=description), training=training)






def _base_cfg():
    cfg = edict()
    cfg.code_version = CODE_VERSION
    cfg.seed = 7351
    cfg.variant = 'full'
    cfg.canonical_variant = 'full'

    cfg.working_dir = osp.dirname(osp.realpath(__file__))
    cfg.root_dir = osp.dirname(osp.dirname(cfg.working_dir))

    cfg.data = edict()
    cfg.data.dataset_root = os.environ.get(
        'PARENET_3DMATCH_ROOT', osp.join(cfg.root_dir, 'datasets', '3DMatch')
    )
    cfg.data.metadata_root = osp.join(cfg.root_dir, 'data', '3DMatch', 'metadata')


    cfg.train = edict()
    cfg.train.batch_size = 1
    cfg.train.num_workers = 0 if os.name == 'nt' else 12
    cfg.train.point_limit = 30000
    cfg.train.use_augmentation = True
    cfg.train.augmentation_noise = 0.005
    cfg.train.augmentation_rotation = 1.0
    cfg.train.augmentation_crop = True
    cfg.train.point_keep_ratio = 0.7
    cfg.train.matching_radius = 0.1

    cfg.test = edict()
    cfg.test.batch_size = 1
    cfg.test.num_workers = 0 if os.name == 'nt' else 8
    cfg.test.point_limit = None
    cfg.test.rotation_mode = 'none'
    cfg.test.rotation_seed = 7351

    cfg.eval = edict()
    cfg.eval.acceptance_overlap = 0.0
    cfg.eval.acceptance_radius = 0.1
    cfg.eval.inlier_ratio_threshold = 0.05
    cfg.eval.rmse_threshold = 0.2
    cfg.eval.rre_threshold = 15.0
    cfg.eval.rte_threshold = 0.3
    cfg.eval.feat_rre_threshold = 20.0

    cfg.ransac = edict()
    cfg.ransac.distance_threshold = 0.05
    cfg.ransac.num_points = 3
    cfg.ransac.num_iterations = 1000

    cfg.optim = edict()
    cfg.optim.lr = 1e-4
    cfg.optim.lr_decay = 0.95
    cfg.optim.lr_decay_steps = 1
    cfg.optim.weight_decay = 1e-6
    cfg.optim.max_epoch = 40
    cfg.optim.grad_acc_steps = 1

    cfg.backbone = edict()
    cfg.backbone.num_stages = 4
    cfg.backbone.num_neighbors = [35] * cfg.backbone.num_stages
    cfg.backbone.init_voxel_size = 0.025
    cfg.backbone.subsample_ratio = 2
    cfg.backbone.kernel_size = 4
    cfg.backbone.share_nonlinearity = False
    cfg.backbone.conv_way = 'edge_conv'
    cfg.backbone.use_xyz = True
    cfg.backbone.init_dim = 96

    cfg.backbone.output_dim = 255
    cfg.backbone.fine_vector_channels = 85


    cfg.backbone.coarse_context = 'spsa'
    cfg.backbone.fine_context = 'mspki'
    cfg.backbone.spsa_partial_ratio = 0.50
    cfg.backbone.spsa_qk_channels = 32
    cfg.backbone.spsa_ffn_ratio = 2.0
    cfg.backbone.mspki_neighbor_scales = (8, 16, 32)
    cfg.backbone.mspki_branch_channels = 16
    cfg.backbone.mspki_ffn_ratio = 2.0
    cfg.backbone.layer_scale_init = 0.10




    cfg.preprocess = edict()
    cfg.preprocess.backend = 'torch_gpu'
    cfg.preprocess.query_chunk_size = 1024
    cfg.preprocess.support_chunk_size = 4096
    cfg.preprocess.require_cuda = True

    cfg.model = edict()
    cfg.model.ground_truth_matching_radius = 0.05
    cfg.model.num_points_in_patch = 64


    cfg.model.estimate_transform_during_training = True

    cfg.coarse_matching = edict()
    cfg.coarse_matching.num_targets = 128
    cfg.coarse_matching.overlap_threshold = 0.1
    cfg.coarse_matching.num_correspondences = 256
    cfg.coarse_matching.dual_normalization = True

    cfg.geotransformer = edict()
    cfg.geotransformer.input_dim = 768
    cfg.geotransformer.hidden_dim = 192
    cfg.geotransformer.output_dim = 192
    cfg.geotransformer.num_heads = 4
    cfg.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
    cfg.geotransformer.sigma_d = 0.1
    cfg.geotransformer.sigma_a = 15
    cfg.geotransformer.angle_k = 3
    cfg.geotransformer.reduction_a = 'max'

    cfg.fine_matching = edict()
    cfg.fine_matching.topk = 3
    cfg.fine_matching.acceptance_radius = 0.1
    cfg.fine_matching.confidence_threshold = 0.005
    cfg.fine_matching.num_hypotheses = 2000
    cfg.fine_matching.num_refinement_steps = 5
    cfg.fine_matching.re_feature_source = 'official'

    cfg.coarse_loss = edict()
    cfg.coarse_loss.positive_margin = 0.1
    cfg.coarse_loss.negative_margin = 1.4
    cfg.coarse_loss.positive_optimal = 0.1
    cfg.coarse_loss.negative_optimal = 1.4
    cfg.coarse_loss.log_scale = 24
    cfg.coarse_loss.positive_overlap = 0.1

    cfg.fine_loss = edict()
    cfg.fine_loss.positive_radius = 0.05
    cfg.fine_loss.negative_radius = 0.2
    cfg.fine_loss.positive_margin = 0.1
    cfg.fine_loss.negative_margin = 1.4

    cfg.loss = edict()
    cfg.loss.weight_coarse_loss = 1.0
    cfg.loss.weight_fine_ri_loss = 1.0
    cfg.loss.weight_fine_re_loss = 1.0



    cfg.paper = edict()
    cfg.paper.spsa_ratios = (0.25, 0.50, 0.75)
    cfg.paper.mspki_support_sets = ((8, 16), (8, 16, 24), (8, 16, 32))
    cfg.paper.layer_scale_inits = (0.01, 0.10, 1.0)
    cfg.paper.density_keep_ratios = (1.00, 0.75, 0.50, 0.25)
    cfg.paper.noise_stds = (0.0, 0.0025, 0.0050, 0.0100)
    cfg.paper.combined_perturbations = ((0.50, 0.005), (0.25, 0.010))
    cfg.paper.fhp_hypothesis_budgets = (100, 250, 500, 1000, 2000)
    cfg.paper.ransac_corr_budgets = (50, 100, 250, 500, None)
    cfg.paper.overlap_bins = ((0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, math.inf))
    cfg.paper.equivariance_num_pairs = 12
    cfg.paper.equivariance_rotations_per_pair = 2
    cfg.paper.equivariance_precisions = ('float32', 'float64')



    return cfg






def _canonical_variant(name: str) -> str:
    aliases = {
        'pki': 'mspki',
        'intended': 'full',
        'pki_coarse': 'mspki_coarse',
    }
    return aliases.get(name, name)


def _apply_variant(cfg):
    cfg.canonical_variant = _canonical_variant(cfg.variant)
    variant = cfg.canonical_variant

    mapping = {
        'baseline': ('none', 'none'),
        'spsa': ('spsa', 'none'),
        'mspki': ('none', 'mspki'),
        'full': ('spsa', 'mspki'),
        'spsa_fine': ('none', 'spsa'),
        'mspki_coarse': ('mspki', 'none'),
        'swapped': ('mspki', 'spsa'),
        'capacity_matched': ('none', 'none'),
    }
    if variant not in mapping:
        raise ValueError(f'Unsupported canonical variant: {variant}')
    cfg.backbone.coarse_context, cfg.backbone.fine_context = mapping[variant]

    if variant == 'capacity_matched':

        cfg.backbone.init_dim = 102
        cfg.backbone.output_dim = 306
        cfg.backbone.fine_vector_channels = 102
        cfg.geotransformer.input_dim = 816
    else:
        cfg.backbone.init_dim = 96
        cfg.backbone.output_dim = 255
        cfg.backbone.fine_vector_channels = 85
        cfg.geotransformer.input_dim = 768


def _format_float_tag(value: float) -> str:
    return format(float(value), 'g').replace('.', 'p').replace('-', 'm')


def _settings_tag(cfg) -> str:
    uses_context = cfg.backbone.coarse_context != 'none' or cfg.backbone.fine_context != 'none'
    if not uses_context:
        return ''



    is_default = (
        abs(float(cfg.backbone.spsa_partial_ratio) - 0.50) < 1e-12
        and int(cfg.backbone.spsa_qk_channels) == 32
        and tuple(cfg.backbone.mspki_neighbor_scales) == (8, 16, 32)
        and abs(float(cfg.backbone.layer_scale_init) - 0.10) < 1e-12
        and int(cfg.backbone.mspki_branch_channels) == 16
        and abs(float(cfg.backbone.spsa_ffn_ratio) - 2.0) < 1e-12
        and abs(float(cfg.backbone.mspki_ffn_ratio) - 2.0) < 1e-12
    )
    if is_default:
        return ''
    scales = '-'.join(str(v) for v in cfg.backbone.mspki_neighbor_scales)
    return (
        f"rho{_format_float_tag(cfg.backbone.spsa_partial_ratio)}_"
        f"qk{cfg.backbone.spsa_qk_channels}_"
        f"ms{scales}_"
        f"ls{_format_float_tag(cfg.backbone.layer_scale_init)}"
    )


def ensure_run_dirs(cfg):
    for path in (
        cfg.output_dir,
        cfg.snapshot_dir,
        cfg.log_dir,
        cfg.event_dir,
        cfg.feature_dir,
        cfg.registration_dir,
    ):
        ensure_dir(path)


def _build_dirs(cfg, output_root=None, run_tag=''):
    if output_root is None:
        output_root = osp.join(cfg.root_dir, 'output', 'indoor_registration_runs')
    cfg.output_root = osp.abspath(osp.expanduser(output_root))

    pieces = [cfg.variant, f"re_{cfg.fine_matching.re_feature_source}", f"seed_{cfg.seed}"]
    settings = _settings_tag(cfg)
    if settings:
        pieces.append(settings)
    if run_tag:
        safe_tag = str(run_tag).strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
        pieces.append(safe_tag)
    cfg.run_name = osp.join(*pieces)
    cfg.exp_name = cfg.run_name
    cfg.output_dir = osp.join(cfg.output_root, cfg.run_name)
    cfg.snapshot_dir = osp.join(cfg.output_dir, 'snapshots')
    cfg.log_dir = osp.join(cfg.output_dir, 'logs')
    cfg.event_dir = osp.join(cfg.output_dir, 'wandb_events')
    cfg.feature_dir = osp.join(cfg.output_dir, 'features')
    cfg.registration_dir = osp.join(cfg.output_dir, 'registration')
    ensure_run_dirs(cfg)


def get_benchmark_tag(cfg, benchmark: str) -> str:
    if cfg.test.rotation_mode == 'none':
        return benchmark
    return f'{benchmark}_{cfg.test.rotation_mode}_seed{cfg.test.rotation_seed}'


def _validate_scales(scales: Sequence[int], max_neighbors: int) -> Tuple[int, ...]:
    values = tuple(sorted({int(v) for v in scales}))
    if not values or values[0] < 1:
        raise ValueError('mspki_scales must contain positive integers.')
    if values[-1] > int(max_neighbors):
        raise ValueError(
            f'Largest MSPKI scale {values[-1]} exceeds available KNN size {max_neighbors}.'
        )
    return values


def make_cfg(args=None):
    cfg = _base_cfg()
    output_root = None
    run_tag = ''

    if args is not None:
        cfg.variant = getattr(args, 'variant', cfg.variant)
        cfg.seed = int(getattr(args, 'seed', cfg.seed))

        if getattr(args, 'dataset_root', None):
            cfg.data.dataset_root = osp.abspath(osp.expanduser(args.dataset_root))
        if getattr(args, 'metadata_root', None):
            cfg.data.metadata_root = osp.abspath(osp.expanduser(args.metadata_root))

        common_workers = getattr(args, 'num_workers', None)
        if common_workers is not None:
            cfg.train.num_workers = int(common_workers)
            cfg.test.num_workers = int(common_workers)
        if getattr(args, 'train_workers', None) is not None:
            cfg.train.num_workers = int(args.train_workers)
        if getattr(args, 'test_workers', None) is not None:
            cfg.test.num_workers = int(args.test_workers)

        cfg.fine_matching.re_feature_source = getattr(
            args, 're_feature_source', cfg.fine_matching.re_feature_source
        )
        cfg.backbone.spsa_partial_ratio = float(
            getattr(args, 'spsa_ratio', cfg.backbone.spsa_partial_ratio)
        )
        cfg.backbone.spsa_qk_channels = int(
            getattr(args, 'spsa_qk_channels', cfg.backbone.spsa_qk_channels)
        )
        cfg.backbone.spsa_ffn_ratio = float(
            getattr(args, 'spsa_ffn_ratio', cfg.backbone.spsa_ffn_ratio)
        )
        cfg.backbone.mspki_neighbor_scales = _validate_scales(
            getattr(args, 'mspki_scales', cfg.backbone.mspki_neighbor_scales),
            cfg.backbone.num_neighbors[1],
        )
        cfg.backbone.mspki_branch_channels = int(
            getattr(args, 'mspki_branch_channels', cfg.backbone.mspki_branch_channels)
        )
        cfg.backbone.mspki_ffn_ratio = float(
            getattr(args, 'mspki_ffn_ratio', cfg.backbone.mspki_ffn_ratio)
        )
        cfg.backbone.layer_scale_init = float(
            getattr(args, 'layer_scale_init', cfg.backbone.layer_scale_init)
        )

        cfg.test.rotation_mode = getattr(args, 'rotation_mode', cfg.test.rotation_mode)
        cfg.test.rotation_seed = int(getattr(args, 'rotation_seed', cfg.test.rotation_seed))
        if getattr(args, 'num_hypotheses', None) is not None:
            cfg.fine_matching.num_hypotheses = int(args.num_hypotheses)
        if getattr(args, 'knn_query_chunk_size', None) is not None:
            cfg.preprocess.query_chunk_size = int(args.knn_query_chunk_size)
        if getattr(args, 'knn_support_chunk_size', None) is not None:
            cfg.preprocess.support_chunk_size = int(args.knn_support_chunk_size)

        if hasattr(args, 'lr'):
            cfg.optim.lr = float(args.lr)
        if hasattr(args, 'max_epoch'):
            cfg.optim.max_epoch = int(args.max_epoch)
        if hasattr(args, 'skip_train_fhp'):
            cfg.model.estimate_transform_during_training = not bool(args.skip_train_fhp)

        output_root = getattr(args, 'output_root', None)
        run_tag = getattr(args, 'run_tag', '') or ''

    cfg.data.dataset_root = osp.abspath(osp.expanduser(cfg.data.dataset_root))
    cfg.data.metadata_root = osp.abspath(osp.expanduser(cfg.data.metadata_root))

    if cfg.variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {cfg.variant}')
    if cfg.fine_matching.re_feature_source not in RE_FEATURE_SOURCES:
        raise ValueError(f'Unknown RE source: {cfg.fine_matching.re_feature_source}')
    if cfg.test.rotation_mode not in ROTATION_MODES:
        raise ValueError(f'Unknown rotation mode: {cfg.test.rotation_mode}')
    if not 0.0 < cfg.backbone.spsa_partial_ratio < 1.0:
        raise ValueError('spsa_ratio must be in (0, 1).')
    if cfg.backbone.spsa_qk_channels < 1:
        raise ValueError('spsa_qk_channels must be positive.')
    if cfg.backbone.spsa_ffn_ratio <= 0 or cfg.backbone.mspki_ffn_ratio <= 0:
        raise ValueError('FFN ratios must be positive.')
    if cfg.backbone.mspki_branch_channels < 1:
        raise ValueError('mspki_branch_channels must be positive.')
    if cfg.backbone.layer_scale_init <= 0:
        raise ValueError('layer_scale_init must be positive.')
    if cfg.fine_matching.num_hypotheses < 1:
        raise ValueError('num_hypotheses must be positive.')
    if cfg.preprocess.query_chunk_size < 1 or cfg.preprocess.support_chunk_size < 1:
        raise ValueError('KNN chunk sizes must be positive.')
    if cfg.train.batch_size != 1 or cfg.test.batch_size != 1:
        raise ValueError('This PARE-Net implementation supports batch_size=1 only.')

    _apply_variant(cfg)
    if cfg.backbone.output_dim % 3 != 0:
        raise ValueError('backbone.output_dim must be divisible by 3.')
    if cfg.geotransformer.input_dim != (8 * cfg.backbone.init_dim // 3) * 3:
        raise ValueError('geotransformer.input_dim is inconsistent with the coarse VN width.')

    _build_dirs(cfg, output_root=output_root, run_tag=run_tag)
    return cfg


if __name__ == '__main__':
    parser = make_experiment_parser(training=True)
    print(make_cfg(parser.parse_args()))
