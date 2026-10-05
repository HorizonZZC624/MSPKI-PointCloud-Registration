from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
PROJECT_ROOT = EXP_DIR.parents[1]
for path in (str(PROJECT_ROOT), str(EXP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.utils.data

from pareconv.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from pareconv.engine import SingleTester
from pareconv.engine.base_tester import inject_default_parser
from pareconv.utils.common import ensure_dir, get_log_string
from pareconv.utils.data import registration_collate_fn_stack_mode, build_dataloader_stack_mode
from pareconv.utils.registration import compute_overlap
from pareconv.utils.torch import release_cuda

from config import get_benchmark_tag, make_cfg, make_experiment_parser
from loss import Evaluator
from model import create_model


def _sha256(file_name, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(file_name, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(payload, file_name):
    temporary = f'{file_name}.tmp'
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temporary, file_name)


def _safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except (AttributeError, KeyError):
        return default


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description='Deterministic paper stress test for PARE-Net/MSPKI.',
    )
    parser.add_argument('--benchmark', default='3DLoMatch', choices=['3DMatch', '3DLoMatch'])
    parser.add_argument('--stress', required=True, choices=['noise', 'density'])
    parser.add_argument('--noise_sigma', type=float, default=0.0,
                        help='Gaussian coordinate noise sigma in meters.')
    parser.add_argument('--density_ratio', type=float, default=1.0,
                        help='Fraction of points retained for density stress.')
    parser.add_argument('--density_target', choices=['src', 'ref', 'both'], default='src',
                        help='Which fragment is downsampled. src is recommended for density mismatch.')
    parser.add_argument('--stress_seed', type=int, default=9101,
                        help='Pair-deterministic perturbation seed shared by all compared models.')
    parser.add_argument('--overlap_radius', type=float, default=0.1,
                        help='Radius used to recompute perturbed pair overlap.')
    parser.add_argument('--overwrite_features', action='store_true')
    return inject_default_parser(parser)


class DeterministicStressDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, stress, noise_sigma, density_ratio,
                 density_target, stress_seed, overlap_radius):
        self.base_dataset = base_dataset
        self.stress = stress
        self.noise_sigma = float(noise_sigma)
        self.density_ratio = float(density_ratio)
        self.density_target = density_target
        self.stress_seed = int(stress_seed)
        self.overlap_radius = float(overlap_radius)

        if self.noise_sigma < 0:
            raise ValueError('--noise_sigma must be >= 0.')
        if not (0 < self.density_ratio <= 1.0):
            raise ValueError('--density_ratio must satisfy 0 < ratio <= 1.')
        if self.overlap_radius <= 0:
            raise ValueError('--overlap_radius must be positive.')

    def __len__(self):
        return len(self.base_dataset)

    def _rng(self, item):
        identity = (
            f"{item['scene_name']}|{item['ref_frame']}|{item['src_frame']}|"
            f"{self.stress}|{self.stress_seed}"
        ).encode('utf-8')
        pair_hash = zlib.crc32(identity) & 0xFFFFFFFF
        return np.random.RandomState((self.stress_seed + pair_hash) % (2**32))

    @staticmethod
    def _sample(points, ratio, rng):
        if ratio >= 1.0 or points.shape[0] <= 1:
            return points
        keep = max(1, int(round(points.shape[0] * ratio)))
        indices = rng.choice(points.shape[0], size=keep, replace=False)
        indices.sort()
        return points[indices]

    def __getitem__(self, index):
        item = self.base_dataset[index]
        rng = self._rng(item)

        ref_points = np.asarray(item['ref_points'], dtype=np.float32).copy()
        src_points = np.asarray(item['src_points'], dtype=np.float32).copy()

        if self.stress == 'noise':
            if self.noise_sigma > 0:
                ref_points += rng.normal(0.0, self.noise_sigma, ref_points.shape).astype(np.float32)
                src_points += rng.normal(0.0, self.noise_sigma, src_points.shape).astype(np.float32)

        elif self.stress == 'density':
            if self.density_target in ('ref', 'both'):
                ref_points = self._sample(ref_points, self.density_ratio, rng)
            if self.density_target in ('src', 'both'):
                src_points = self._sample(src_points, self.density_ratio, rng)

        item['ref_points'] = np.ascontiguousarray(ref_points, dtype=np.float32)
        item['src_points'] = np.ascontiguousarray(src_points, dtype=np.float32)
        item['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        item['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        item['overlap'] = float(
            compute_overlap(
                ref_points,
                src_points,
                np.asarray(item['transform']),
                positive_radius=self.overlap_radius,
            )
        )
        return item


def build_stress_loader(cfg, args):
    rotation_mode = _safe_attr(cfg.test, 'rotation_mode', 'none')
    rotation_seed = int(_safe_attr(cfg.test, 'rotation_seed', 7351))
    base_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        args.benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
        rotated=(rotation_mode == 'so3'),
        rotation_seed=rotation_seed,
    )
    dataset = DeterministicStressDataset(
        base_dataset,
        stress=args.stress,
        noise_sigma=args.noise_sigma,
        density_ratio=args.density_ratio,
        density_target=args.density_target,
        stress_seed=args.stress_seed,
        overlap_radius=args.overlap_radius,
    )
    loader = build_dataloader_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        precompute_data=False,
    )
    return loader, cfg.backbone.num_neighbors


class StressTester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        start = time.time()
        data_loader, neighbor_limits = build_stress_loader(cfg, self.args)
        self.logger.info(f'Stress data loader created in {time.time() - start:.3f}s.')
        self.logger.info(f'KNN limits: {neighbor_limits}.')
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.total_params = sum(p.numel() for p in model.parameters())
        self.logger.critical(
            f'Variant={cfg.variant}, params={self.total_params:,}, '
            f'RE={cfg.fine_matching.re_feature_source}, rotation={cfg.test.rotation_mode}, '
            f'stress={self.args.stress}, noise_sigma={self.args.noise_sigma}, '
            f'density_ratio={self.args.density_ratio}, density_target={self.args.density_target}, '
            f'stress_seed={self.args.stress_seed}'
        )
        self.register_model(model)
        self.evaluator = Evaluator(cfg).cuda()

        self.benchmark_tag = get_benchmark_tag(cfg, self.args.benchmark)
        self.output_dir = osp.join(cfg.feature_dir, self.benchmark_tag)
        self.registration_output_dir = osp.join(cfg.registration_dir, self.benchmark_tag)
        if osp.isdir(self.output_dir):
            existing = list(Path(self.output_dir).rglob('*.npz'))
            manifest_exists = osp.isfile(osp.join(self.output_dir, 'manifest.json'))
            if (existing or manifest_exists) and not self.args.overwrite_features:
                raise FileExistsError(
                    f'{self.output_dir} already contains extracted output. Use --overwrite_features.'
                )
            if self.args.overwrite_features:
                shutil.rmtree(self.output_dir)
        if self.args.overwrite_features and osp.isdir(self.registration_output_dir):
            shutil.rmtree(self.registration_output_dir)
        ensure_dir(self.output_dir)
        self.manifest_file = osp.join(self.output_dir, 'manifest.json')
        self.saved_pairs = 0
        self.expected_pairs = len(self.test_loader.dataset)

    def before_test_epoch(self):
        checkpoint_seed = int(self.checkpoint_metadata.get('seed', self.cfg.seed))
        self.manifest = {
            'code_version': _safe_attr(self.cfg, 'code_version'),
            'checkpoint_code_version': self.checkpoint_metadata.get('code_version'),
            'variant': self.cfg.variant,
            'canonical_variant': _safe_attr(self.cfg, 'canonical_variant', self.cfg.variant),
            'seed': checkpoint_seed,
            're_feature_source': self.cfg.fine_matching.re_feature_source,
            'coarse_context': _safe_attr(self.cfg.backbone, 'coarse_context'),
            'fine_context': _safe_attr(self.cfg.backbone, 'fine_context'),
            'mspki_scales': list(_safe_attr(self.cfg.backbone, 'mspki_neighbor_scales', [])),
            'num_hypotheses': _safe_attr(self.cfg.fine_matching, 'num_hypotheses'),
            'benchmark': self.args.benchmark,
            'benchmark_tag': self.benchmark_tag,
            'rotation_mode': _safe_attr(self.cfg.test, 'rotation_mode', 'none'),
            'rotation_seed': int(_safe_attr(self.cfg.test, 'rotation_seed', 7351)),
            'stress_type': self.args.stress,
            'noise_sigma_m': float(self.args.noise_sigma),
            'density_ratio': float(self.args.density_ratio),
            'density_target': self.args.density_target,
            'stress_seed': int(self.args.stress_seed),
            'overlap_radius_m': float(self.args.overlap_radius),
            'snapshot': osp.abspath(self.args.snapshot),
            'snapshot_sha256': _sha256(self.args.snapshot),
            'parameters': int(self.total_params),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'expected_pairs': int(self.expected_pairs),
            'saved_pairs': 0,
            'completed': False,
        }
        _atomic_json_dump(self.manifest, self.manifest_file)

    def test_step(self, iteration, data_dict):
        return self.model(data_dict, compute_gt=True, estimate_transform=True)

    def eval_step(self, iteration, data_dict, output_dict):
        return self.evaluator(output_dict, data_dict)

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        del iteration, output_dict
        message = (
            f"{data_dict['scene_name']}, id0={data_dict['ref_frame']}, "
            f"id1={data_dict['src_frame']}"
        )
        return message + ', ' + get_log_string(result_dict=result_dict)

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        del iteration, result_dict
        scene_dir = osp.join(self.output_dir, data_dict['scene_name'])
        ensure_dir(scene_dir)
        file_name = osp.join(scene_dir, f"{data_dict['ref_frame']}_{data_dict['src_frame']}.npz")
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            hypotheses=release_cuda(output_dict['hypotheses']),
            hypothesis_sources=release_cuda(output_dict['hypothesis_sources']),
            registration_attempted=release_cuda(output_dict['registration_attempted']),
            registration_succeeded=release_cuda(output_dict['registration_succeeded']),
            transform=release_cuda(data_dict['transform']),
            overlap=release_cuda(data_dict['overlap']),
        )
        self.saved_pairs += 1

    def after_test_epoch(self):
        self.manifest['saved_pairs'] = int(self.saved_pairs)
        self.manifest['completed'] = bool(self.saved_pairs == self.expected_pairs)
        self.manifest['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _atomic_json_dump(self.manifest, self.manifest_file)
        if not self.manifest['completed']:
            raise RuntimeError(
                f'Extraction incomplete: saved={self.saved_pairs}, expected={self.expected_pairs}.'
            )


def main():
    parser = make_parser()
    known_args, _ = parser.parse_known_args()
    cfg = make_cfg(known_args)


    if getattr(cfg.test, 'rotation_mode', 'none') != 'none':
        print('[WARNING] rotation_mode is not none; stress and SO(3) are being combined.')


    random.seed(known_args.stress_seed)
    np.random.seed(known_args.stress_seed)
    torch.manual_seed(known_args.stress_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(known_args.stress_seed)

    StressTester(cfg, parser).run()


if __name__ == '__main__':
    main()
