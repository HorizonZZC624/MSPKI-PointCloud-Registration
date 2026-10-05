
from __future__ import annotations

import argparse
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

from pareconv.utils.common import ensure_dir

CODE_VERSION = "kitti-mspki-official-v2-fair"
VARIANTS = ("baseline", "mspki")
RE_FEATURE_SOURCES = ("official", "decoder", "encoder")


def make_experiment_parser(training: bool = False, description: Optional[str] = None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--seed", type=int, default=7351)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--metadata_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--re_feature_source", choices=RE_FEATURE_SOURCES, default="official")

    p.add_argument("--mspki_scales", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--mspki_branch_channels", type=int, default=16)
    p.add_argument("--mspki_ffn_ratio", type=float, default=2.0)
    p.add_argument("--layer_scale_init", type=float, default=0.10)

    p.add_argument("--knn_query_chunk_size", type=int, default=1024)
    p.add_argument("--knn_support_chunk_size", type=int, default=4096)

    if training:
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--max_epoch", type=int, default=100)
        p.add_argument(
            "--skip_train_fhp",
            action="store_true",
            help="Engineering speed option only. FHP is no_grad and is not part of the loss.",
        )
        p.add_argument("--overwrite_run", action="store_true")
    return p


def _base_cfg():
    cfg = edict()
    cfg.code_version = CODE_VERSION
    cfg.seed = 7351
    cfg.variant = "baseline"
    cfg.canonical_variant = "baseline"

    cfg.working_dir = osp.dirname(osp.realpath(__file__))
    cfg.root_dir = osp.dirname(osp.dirname(cfg.working_dir))

    cfg.data = edict()
    cfg.data.dataset_root = osp.join(cfg.root_dir, "datasets", "KITTI")
    cfg.data.metadata_root = osp.join(cfg.root_dir, "data", "KITTI", "metadata")


    cfg.train = edict()
    cfg.train.batch_size = 1
    cfg.train.num_workers = 8
    cfg.train.point_limit = 30000
    cfg.train.use_augmentation = True
    cfg.train.augmentation_noise = 0.01
    cfg.train.augmentation_min_scale = 0.8
    cfg.train.augmentation_max_scale = 1.1
    cfg.train.augmentation_shift = 2.0
    cfg.train.augmentation_rotation = 1.0
    cfg.train.augmentation_crop = True
    cfg.train.point_keep_ratio = 0.7
    cfg.train.matching_radius = 0.6

    cfg.test = edict()
    cfg.test.batch_size = 1
    cfg.test.num_workers = 8
    cfg.test.point_limit = None

    cfg.eval = edict()
    cfg.eval.acceptance_overlap = 0.0
    cfg.eval.acceptance_radius = 1.0
    cfg.eval.inlier_ratio_threshold = 0.05
    cfg.eval.rre_threshold = 5.0
    cfg.eval.rte_threshold = 2.0
    cfg.eval.feat_rre_threshold = 20.0

    cfg.ransac = edict()
    cfg.ransac.distance_threshold = 0.3
    cfg.ransac.num_points = 4
    cfg.ransac.num_iterations = 5000

    cfg.optim = edict()
    cfg.optim.lr = 1e-4
    cfg.optim.lr_decay = 0.95
    cfg.optim.lr_decay_steps = 4
    cfg.optim.weight_decay = 1e-6
    cfg.optim.max_epoch = 100
    cfg.optim.grad_acc_steps = 1

    cfg.backbone = edict()
    cfg.backbone.num_stages = 4
    cfg.backbone.num_neighbors = [35] * cfg.backbone.num_stages
    cfg.backbone.init_voxel_size = 0.3
    cfg.backbone.subsample_ratio = 2.5
    cfg.backbone.kernel_size = 4
    cfg.backbone.share_nonlinearity = False
    cfg.backbone.conv_way = "edge_conv"
    cfg.backbone.use_xyz = True
    cfg.backbone.init_dim = 96
    cfg.backbone.output_dim = 64


    cfg.backbone.coarse_context = "none"
    cfg.backbone.fine_context = "none"
    cfg.backbone.spsa_partial_ratio = 0.5
    cfg.backbone.spsa_qk_channels = 32
    cfg.backbone.spsa_ffn_ratio = 2.0
    cfg.backbone.mspki_neighbor_scales = (8, 16, 32)
    cfg.backbone.mspki_branch_channels = 16
    cfg.backbone.mspki_ffn_ratio = 2.0
    cfg.backbone.layer_scale_init = 0.10


    cfg.preprocess = edict()
    cfg.preprocess.query_chunk_size = 1024
    cfg.preprocess.support_chunk_size = 4096
    cfg.preprocess.require_cuda = True

    cfg.model = edict()
    cfg.model.ground_truth_matching_radius = 0.6
    cfg.model.num_points_in_patch = 128
    cfg.model.estimate_transform_during_training = True

    cfg.coarse_matching = edict()
    cfg.coarse_matching.num_targets = 128
    cfg.coarse_matching.overlap_threshold = 0.1
    cfg.coarse_matching.num_correspondences = 256
    cfg.coarse_matching.dual_normalization = True

    cfg.geotransformer = edict()
    cfg.geotransformer.input_dim = 768
    cfg.geotransformer.hidden_dim = 96
    cfg.geotransformer.output_dim = 128
    cfg.geotransformer.num_heads = 4
    cfg.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
    cfg.geotransformer.sigma_d = 4.8
    cfg.geotransformer.sigma_a = 15
    cfg.geotransformer.angle_k = 3
    cfg.geotransformer.reduction_a = "max"

    cfg.fine_matching = edict()
    cfg.fine_matching.topk = 2
    cfg.fine_matching.acceptance_radius = 0.6
    cfg.fine_matching.confidence_threshold = 0.005
    cfg.fine_matching.num_hypotheses = 1000
    cfg.fine_matching.num_refinement_steps = 5
    cfg.fine_matching.re_feature_source = "official"

    cfg.coarse_loss = edict()
    cfg.coarse_loss.positive_margin = 0.1
    cfg.coarse_loss.negative_margin = 1.4
    cfg.coarse_loss.positive_optimal = 0.1
    cfg.coarse_loss.negative_optimal = 1.4
    cfg.coarse_loss.log_scale = 40
    cfg.coarse_loss.positive_overlap = 0.1

    cfg.fine_loss = edict()
    cfg.fine_loss.positive_radius = 0.6
    cfg.fine_loss.negative_radius = 1.2
    cfg.fine_loss.positive_margin = 0.1
    cfg.fine_loss.negative_margin = 1.4

    cfg.loss = edict()
    cfg.loss.weight_coarse_loss = 1.0
    cfg.loss.weight_fine_ri_loss = 1.0
    cfg.loss.weight_fine_re_loss = 1.0
    return cfg


def _validate_scales(scales: Sequence[int], max_neighbors: int) -> Tuple[int, ...]:
    values = tuple(sorted({int(v) for v in scales}))
    if not values or values[0] < 1:
        raise ValueError("mspki_scales must contain positive integers.")
    if values[-1] > int(max_neighbors):
        raise ValueError(
            f"Largest MSPKI scale {values[-1]} exceeds available KNN size {max_neighbors}."
        )
    return values


def _apply_variant(cfg):
    if cfg.variant == "baseline":
        cfg.canonical_variant = "baseline"
        cfg.backbone.coarse_context = "none"
        cfg.backbone.fine_context = "none"
    elif cfg.variant == "mspki":
        cfg.canonical_variant = "mspki"
        cfg.backbone.coarse_context = "none"
        cfg.backbone.fine_context = "mspki"
    else:
        raise ValueError(cfg.variant)


def _build_dirs(cfg, output_root=None, run_tag=""):
    if output_root is None:
        output_root = osp.join(cfg.root_dir, "output", "KITTI_mspki")
    cfg.output_root = osp.abspath(osp.expanduser(output_root))

    pieces = [cfg.variant, f"re_{cfg.fine_matching.re_feature_source}", f"seed_{cfg.seed}"]
    if run_tag:
        pieces.append(str(run_tag).strip().replace(" ", "_").replace("/", "_"))
    cfg.run_name = osp.join(*pieces)
    cfg.output_dir = osp.join(cfg.output_root, cfg.run_name)
    cfg.snapshot_dir = osp.join(cfg.output_dir, "snapshots")
    cfg.log_dir = osp.join(cfg.output_dir, "logs")
    cfg.event_dir = osp.join(cfg.output_dir, "wandb_events")
    cfg.feature_dir = osp.join(cfg.output_dir, "features")
    cfg.registration_dir = osp.join(cfg.output_dir, "registration")

    for p in (
        cfg.output_dir, cfg.snapshot_dir, cfg.log_dir, cfg.event_dir,
        cfg.feature_dir, cfg.registration_dir,
    ):
        ensure_dir(p)


def make_cfg(args=None):
    cfg = _base_cfg()
    output_root = None
    run_tag = ""

    if args is not None:
        cfg.variant = getattr(args, "variant", cfg.variant)
        cfg.seed = int(getattr(args, "seed", cfg.seed))
        if getattr(args, "dataset_root", None):
            cfg.data.dataset_root = osp.abspath(osp.expanduser(args.dataset_root))
        if getattr(args, "metadata_root", None):
            cfg.data.metadata_root = osp.abspath(osp.expanduser(args.metadata_root))
        if getattr(args, "output_root", None):
            output_root = args.output_root
        run_tag = getattr(args, "run_tag", "")

        if getattr(args, "num_workers", None) is not None:
            cfg.train.num_workers = int(args.num_workers)
            cfg.test.num_workers = int(args.num_workers)

        cfg.fine_matching.re_feature_source = getattr(
            args, "re_feature_source", cfg.fine_matching.re_feature_source
        )
        cfg.backbone.mspki_neighbor_scales = _validate_scales(
            getattr(args, "mspki_scales", cfg.backbone.mspki_neighbor_scales),
            cfg.backbone.num_neighbors[1],
        )
        cfg.backbone.mspki_branch_channels = int(
            getattr(args, "mspki_branch_channels", cfg.backbone.mspki_branch_channels)
        )
        cfg.backbone.mspki_ffn_ratio = float(
            getattr(args, "mspki_ffn_ratio", cfg.backbone.mspki_ffn_ratio)
        )
        cfg.backbone.layer_scale_init = float(
            getattr(args, "layer_scale_init", cfg.backbone.layer_scale_init)
        )

        cfg.preprocess.query_chunk_size = int(
            getattr(args, "knn_query_chunk_size", cfg.preprocess.query_chunk_size)
        )
        cfg.preprocess.support_chunk_size = int(
            getattr(args, "knn_support_chunk_size", cfg.preprocess.support_chunk_size)
        )

        if hasattr(args, "lr"):
            cfg.optim.lr = float(args.lr)
        if hasattr(args, "max_epoch"):
            cfg.optim.max_epoch = int(args.max_epoch)
        if hasattr(args, "skip_train_fhp") and args.skip_train_fhp:
            cfg.model.estimate_transform_during_training = False

    _apply_variant(cfg)
    _build_dirs(cfg, output_root=output_root, run_tag=run_tag)

    if cfg.fine_matching.re_feature_source not in RE_FEATURE_SOURCES:
        raise ValueError(cfg.fine_matching.re_feature_source)
    return cfg
