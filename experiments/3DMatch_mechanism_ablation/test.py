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

from config import get_benchmark_tag, make_cfg, make_experiment_parser
from dataset import test_data_loader
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


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description='Extract correspondences for official/rotated PARE-Net evaluation.',
    )
    parser.add_argument('--benchmark', default='3DMatch', choices=['3DMatch', '3DLoMatch', 'val'])
    parser.add_argument('--overwrite_features', action='store_true')
    return inject_default_parser(parser)


class Tester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        start = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        self.logger.info(f'Data loader created in {time.time() - start:.3f}s.')
        self.logger.info(f'KNN limits: {neighbor_limits}.')
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.total_params = sum(p.numel() for p in model.parameters())
        self.logger.critical(
            f'Variant={cfg.variant}, canonical={cfg.canonical_variant}, params={self.total_params:,}, '
            f'coarse={cfg.backbone.coarse_context}, fine={cfg.backbone.fine_context}, '
            f'RE={cfg.fine_matching.re_feature_source}, rotation={cfg.test.rotation_mode}'
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




        checkpoint_cfg = self.checkpoint_metadata.get('config') or {}
        checkpoint_backbone = checkpoint_cfg.get('backbone') or {}
        checkpoint_fusion = checkpoint_backbone.get('mspki_fusion', 'learned')
        checkpoint_scales = tuple(
            checkpoint_backbone.get('mspki_neighbor_scales', self.cfg.backbone.mspki_neighbor_scales)
        )
        if str(checkpoint_fusion) != str(self.cfg.backbone.mspki_fusion):
            raise ValueError(
                f'Checkpoint MSPKI fusion={checkpoint_fusion!r} does not match '
                f'--mspki_fusion={self.cfg.backbone.mspki_fusion!r}.'
            )
        if checkpoint_scales != tuple(self.cfg.backbone.mspki_neighbor_scales):
            raise ValueError(
                f'Checkpoint MSPKI scales={checkpoint_scales!r} do not match '
                f'--mspki_scales={tuple(self.cfg.backbone.mspki_neighbor_scales)!r}.'
            )

        checkpoint_seed = int(self.checkpoint_metadata.get('seed', self.cfg.seed))
        self.manifest = {
            'code_version': self.cfg.code_version,
            'checkpoint_code_version': self.checkpoint_metadata.get('code_version'),
            'variant': self.cfg.variant,
            'canonical_variant': self.cfg.canonical_variant,
            'seed': checkpoint_seed,
            're_feature_source': self.cfg.fine_matching.re_feature_source,
            'coarse_context': self.cfg.backbone.coarse_context,
            'fine_context': self.cfg.backbone.fine_context,
            'spsa_ratio': self.cfg.backbone.spsa_partial_ratio,
            'mspki_scales': list(self.cfg.backbone.mspki_neighbor_scales),
            'mspki_fusion': self.cfg.backbone.mspki_fusion,
            'layer_scale_init': self.cfg.backbone.layer_scale_init,
            'num_hypotheses': self.cfg.fine_matching.num_hypotheses,
            'benchmark': self.args.benchmark,
            'benchmark_tag': self.benchmark_tag,
            'rotation_mode': self.cfg.test.rotation_mode,
            'rotation_seed': self.cfg.test.rotation_seed,
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
    Tester(cfg, parser).run()


if __name__ == '__main__':
    main()
