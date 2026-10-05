from __future__ import annotations

import argparse
import csv
import glob
import json
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
PROJECT_ROOT = EXP_DIR.parents[1]
for path in (str(PROJECT_ROOT), str(EXP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from pareconv.datasets.registration.threedmatch.utils import (
    compute_transform_error,
    get_gt_logs_and_infos,
    get_num_fragments,
)
from pareconv.utils.registration import (
    compute_registration_error,
    evaluate_correspondences,
    evaluate_sparse_correspondences,
)


def _is_valid_rigid_transform(transform):
    transform = np.asarray(transform)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4, rtol=0):
        return False
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2, rtol=0):
        return False
    return abs(np.linalg.det(rotation) - 1.0) < 1e-2


def _read_manifest(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _find_features_root(search_root, benchmark, variant, seed, re_feature_source):
    matches = []
    for manifest_path in Path(search_root).rglob('manifest.json'):
        manifest = _read_manifest(manifest_path)
        if not manifest:
            continue
        if manifest.get('benchmark') != benchmark:
            continue
        if manifest.get('variant') != variant:
            continue
        if int(manifest.get('seed', -1)) != int(seed):
            continue
        if re_feature_source is not None and manifest.get('re_feature_source') != re_feature_source:
            continue
        if manifest.get('rotation_mode', 'none') != 'none':
            continue
        if not manifest.get('completed', False):
            continue
        matches.append((manifest_path.parent, manifest))

    if len(matches) != 1:
        details = '\n'.join(str(item[0]) for item in matches)
        raise RuntimeError(
            f'Expected exactly one matching completed feature root, found {len(matches)}.\n{details}'
        )
    return matches[0]


def _default_bins(benchmark):
    if benchmark == '3DLoMatch':
        return [0.0, 0.15, 0.20, 0.25, 0.30, 1.01]
    return [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 1.01]


def _parse_bins(text, benchmark):
    if text is None:
        return _default_bins(benchmark)
    values = [float(x.strip()) for x in text.split(',') if x.strip()]
    if len(values) < 2 or values != sorted(values):
        raise ValueError('--bins must be an increasing comma-separated sequence.')
    return values


def _bin_index(value, bins):
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if lo <= value < hi:
            return i
    if np.isclose(value, bins[-1]):
        return len(bins) - 2
    return None


def _label(lo, hi):
    if hi > 1.0:
        return f'[{lo:.2f},1.00]'
    return f'[{lo:.2f},{hi:.2f})'


def _mean(values):
    return float(np.mean(values)) if values else float('nan')


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else float('nan')


def main():
    parser = argparse.ArgumentParser(description='Overlap-stratified official 3DMatch/3DLoMatch evaluation.')
    parser.add_argument('--search_root', required=True,
                        help='Top-level output root containing completed test features.')
    parser.add_argument('--benchmark', required=True, choices=['3DMatch', '3DLoMatch'])
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--re_feature_source', default='official')
    parser.add_argument('--metadata_root', default=None)
    parser.add_argument('--bins', default=None,
                        help='Comma-separated edges. Defaults are benchmark-specific.')
    parser.add_argument('--inlier_radius', type=float, default=0.1)
    parser.add_argument('--fmr_threshold', type=float, default=0.05)
    parser.add_argument('--rmse_threshold', type=float, default=0.2)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if args.metadata_root is None:
        args.metadata_root = str(PROJECT_ROOT / 'data' / '3DMatch' / 'metadata')

    features_root, manifest = _find_features_root(
        args.search_root,
        args.benchmark,
        args.variant,
        args.seed,
        args.re_feature_source,
    )
    bins = _parse_bins(args.bins, args.benchmark)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pair_rows = []
    scene_bin = defaultdict(lambda: defaultdict(lambda: {
        'pir': [], 'fmr': [], 'ir': [], 'rr_success': [], 'rre': [], 'rte': []
    }))

    scene_dirs = sorted(p for p in features_root.iterdir() if p.is_dir())
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        num_fragments = get_num_fragments(scene_name)
        gt_root = osp.join(args.metadata_root, 'benchmarks', args.benchmark, scene_name)
        gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_root, num_fragments)

        for file_name in sorted(glob.glob(osp.join(scene_dir, '*.npz'))):
            ref_frame, src_frame = [int(x) for x in osp.basename(file_name).split('.')[0].split('_')]
            with np.load(file_name) as data:
                overlap = float(np.asarray(data['overlap']).item())
                bi = _bin_index(overlap, bins)
                if bi is None:
                    continue

                coarse = evaluate_sparse_correspondences(
                    data['ref_points_c'], data['src_points_c'],
                    data['ref_node_corr_indices'], data['src_node_corr_indices'],
                    data['gt_node_corr_indices'],
                )
                pir = float(coarse['precision'])

                ref_corr = data['ref_corr_points']
                src_corr = data['src_corr_points']
                if ref_corr.shape[0] == 0:
                    ir = 0.0
                else:
                    fine = evaluate_correspondences(
                        ref_corr, src_corr, data['transform'], positive_radius=args.inlier_radius
                    )
                    ir = float(fine['inlier_ratio'])
                fmr = float(ir >= args.fmr_threshold)

                gt_index = int(gt_indices[ref_frame, src_frame])
                is_gt_pair = gt_index != -1
                rr_success = None
                rre = None
                rte = None
                if is_gt_pair:
                    succeeded = bool(np.asarray(data['registration_succeeded']).item())
                    estimated = np.asarray(data['estimated_transform'])
                    succeeded = succeeded and _is_valid_rigid_transform(estimated)
                    rr_success = 0.0
                    if succeeded:
                        gt_transform = gt_logs[gt_index]['transform']
                        covariance = gt_infos[gt_index]['covariance']
                        err = compute_transform_error(gt_transform, covariance, estimated)
                        if np.isfinite(err) and err <= args.rmse_threshold ** 2:
                            rr_success = 1.0
                            rre, rte = compute_registration_error(gt_transform, estimated)
                            rre, rte = float(rre), float(rte)

                bucket = scene_bin[scene_name][bi]
                bucket['pir'].append(pir)
                bucket['fmr'].append(fmr)
                bucket['ir'].append(ir)
                if rr_success is not None:
                    bucket['rr_success'].append(rr_success)
                if rre is not None:
                    bucket['rre'].append(rre)
                    bucket['rte'].append(rte)

                pair_rows.append({
                    'scene': scene_name,
                    'ref_frame': ref_frame,
                    'src_frame': src_frame,
                    'overlap': overlap,
                    'bin': _label(bins[bi], bins[bi + 1]),
                    'PIR': pir,
                    'FMR': fmr,
                    'IR': ir,
                    'is_official_gt_pair': int(is_gt_pair),
                    'RR_success': '' if rr_success is None else rr_success,
                    'RRE': '' if rre is None else rre,
                    'RTE_m': '' if rte is None else rte,
                })

    pair_csv = Path(args.output_dir) / f'{args.variant}_seed{args.seed}_{args.benchmark}_overlap_pairs.csv'
    with open(pair_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else ['scene'])
        writer.writeheader()
        writer.writerows(pair_rows)

    summary_rows = []
    for bi in range(len(bins) - 1):
        pair_metrics = {'pir': [], 'fmr': [], 'ir': [], 'rr': [], 'rre': [], 'rte': []}
        scene_metrics = {'pir': [], 'fmr': [], 'ir': [], 'rr': [], 'rre': [], 'rte': []}
        pair_count = 0
        gt_pair_count = 0

        for scene_name, bins_dict in scene_bin.items():
            b = bins_dict.get(bi)
            if not b:
                continue
            pair_count += len(b['ir'])
            gt_pair_count += len(b['rr_success'])
            pair_metrics['pir'].extend(b['pir'])
            pair_metrics['fmr'].extend(b['fmr'])
            pair_metrics['ir'].extend(b['ir'])
            pair_metrics['rr'].extend(b['rr_success'])
            pair_metrics['rre'].extend(b['rre'])
            pair_metrics['rte'].extend(b['rte'])

            scene_metrics['pir'].append(_mean(b['pir']))
            scene_metrics['fmr'].append(_mean(b['fmr']))
            scene_metrics['ir'].append(_mean(b['ir']))
            if b['rr_success']:
                scene_metrics['rr'].append(_mean(b['rr_success']))
            if b['rre']:
                scene_metrics['rre'].append(_mean(b['rre']))
                scene_metrics['rte'].append(_mean(b['rte']))

        row = {
            'variant': args.variant,
            'seed': args.seed,
            'benchmark': args.benchmark,
            'bin': _label(bins[bi], bins[bi + 1]),
            'bin_low': bins[bi],
            'bin_high': bins[bi + 1],
            'pair_count': pair_count,
            'official_gt_pair_count': gt_pair_count,
            'PIR_micro': _mean(pair_metrics['pir']),
            'FMR_micro': _mean(pair_metrics['fmr']),
            'IR_micro': _mean(pair_metrics['ir']),
            'RR_micro': _mean(pair_metrics['rr']),
            'RRE_success_micro': _mean(pair_metrics['rre']),
            'RTE_success_micro_m': _mean(pair_metrics['rte']),
            'PIR_macro_scene': _mean(scene_metrics['pir']),
            'FMR_macro_scene': _mean(scene_metrics['fmr']),
            'IR_macro_scene': _mean(scene_metrics['ir']),
            'RR_macro_scene': _mean(scene_metrics['rr']),
            'RRE_success_macro_scene': _mean(scene_metrics['rre']),
            'RTE_success_macro_scene_m': _mean(scene_metrics['rte']),
        }
        summary_rows.append(row)

    summary_csv = Path(args.output_dir) / f'{args.variant}_seed{args.seed}_{args.benchmark}_overlap_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = Path(args.output_dir) / f'{args.variant}_seed{args.seed}_{args.benchmark}_overlap_summary.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump({
            'manifest': manifest,
            'features_root': str(features_root),
            'bins': bins,
            'summary': summary_rows,
        }, f, indent=2, ensure_ascii=False, allow_nan=True)

    print(f'Features: {features_root}')
    print(f'Pair CSV: {pair_csv}')
    print(f'Summary CSV: {summary_csv}')
    for row in summary_rows:
        print(
            f"{row['bin']}: n={row['pair_count']}, "
            f"IR_macro={row['IR_macro_scene']:.4f}, RR_macro={row['RR_macro_scene']:.4f}"
        )


if __name__ == '__main__':
    main()
