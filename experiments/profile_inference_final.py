from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import platform
import subprocess
import sys
import time
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
from pareconv.utils.data import registration_collate_fn_stack_mode, build_dataloader_stack_mode

from config import make_cfg, make_experiment_parser
from model import create_model


PROFILE_PROTOCOL_VERSION = 'paper-inference-only-v2'


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _command_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def get_driver_version():
    out = _command_output(
        [
            'nvidia-smi',
            '--query-gpu=driver_version',
            '--format=csv,noheader',
        ]
    )
    if not out:
        return None
    return out.splitlines()[0].strip()


def get_git_commit():
    return _command_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'])


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description='Profile PARE-Net/MSPKI inference-only forward latency.',
    )
    parser.add_argument('--benchmark', default='3DLoMatch', choices=['3DMatch', '3DLoMatch'])
    parser.add_argument('--profile_pairs', type=int, default=100)
    parser.add_argument('--warmup_pairs', type=int, default=20)
    parser.add_argument('--save_json', required=True)
    return inject_default_parser(parser)


def build_profile_loader(cfg, benchmark, total_pairs):
    dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
        rotated=False,
    )
    total_pairs = min(total_pairs, len(dataset))
    dataset = torch.utils.data.Subset(dataset, range(total_pairs))
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
    return loader


class ProfileTester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        total = self.args.warmup_pairs + self.args.profile_pairs
        loader = build_profile_loader(cfg, self.args.benchmark, total)
        self.register_loader(loader)

        model = create_model(cfg).cuda()
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.register_model(model)

        self.times_ms = []
        self._timed_memory_reset_done = False

    def before_test_epoch(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def test_step(self, iteration, data_dict):





        if (
            not self._timed_memory_reset_done
            and iteration == self.args.warmup_pairs + 1
        ):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self._timed_memory_reset_done = True

        torch.cuda.synchronize()
        start = time.perf_counter()




        output = self.model(
            data_dict,
            compute_gt=False,
            estimate_transform=True,
        )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0


        if iteration > self.args.warmup_pairs:
            self.times_ms.append(elapsed_ms)

        return output

    def eval_step(self, iteration, data_dict, output_dict):
        return {}

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        del data_dict, output_dict, result_dict
        phase = 'warmup' if iteration <= self.args.warmup_pairs else 'timed'
        return f'profile iteration={iteration}, phase={phase}'

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        del iteration, data_dict, output_dict, result_dict

    def after_test_epoch(self):
        times = self.times_ms
        if not times:
            raise RuntimeError('No timed profile pairs were recorded.')
        if len(times) != self.args.profile_pairs:
            raise RuntimeError(
                f'Expected {self.args.profile_pairs} timed pairs, but recorded {len(times)}. '
                'Check dataset size / warmup settings.'
            )

        device_idx = torch.cuda.current_device()
        device = torch.cuda.get_device_properties(device_idx)

        snapshot_abs = osp.abspath(self.args.snapshot)

        result = {
            'profile_protocol_version': PROFILE_PROTOCOL_VERSION,
            'variant': self.cfg.variant,
            'seed': self.cfg.seed,
            'benchmark': self.args.benchmark,
            're_feature_source': self.cfg.fine_matching.re_feature_source,
            'snapshot': snapshot_abs,
            'snapshot_sha256': _sha256(snapshot_abs),
            'git_commit': get_git_commit(),
            'parameters': int(self.total_params),
            'trainable_parameters': int(self.trainable_params),
            'warmup_pairs': int(self.args.warmup_pairs),
            'profile_pairs': len(times),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'pair_latencies_ms': [float(x) for x in times],
            'peak_gpu_memory_mib': float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            'peak_gpu_reserved_mib': float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            'timing_scope': 'model_forward_plus_pose_estimation',
            'data_loading_included': False,
            'host_to_device_transfer_included': False,
            'ground_truth_computation_included': False,
            'pose_estimation_included': True,
            'gpu_logical_index': int(device_idx),
            'gpu_name': torch.cuda.get_device_name(device_idx),
            'gpu_total_memory_gib': float(device.total_memory / (1024 ** 3)),
            'compute_capability': f'{device.major}.{device.minor}',
            'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_runtime_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'nvidia_driver_version': get_driver_version(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }

        Path(self.args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(self.args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = make_parser()
    known_args, _ = parser.parse_known_args()
    if known_args.profile_pairs < 1 or known_args.warmup_pairs < 0:
        raise ValueError('profile_pairs must be >=1 and warmup_pairs >=0.')
    cfg = make_cfg(known_args)
    ProfileTester(cfg, parser).run()


if __name__ == '__main__':
    main()
