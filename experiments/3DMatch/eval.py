import csv
import glob
import json
import os.path as osp
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from pareconv.datasets.registration.threedmatch.utils import (
    compute_transform_error,
    get_gt_logs_and_infos,
    get_num_fragments,
    get_scene_abbr,
    write_log_file,
)
from pareconv.engine import Logger
from pareconv.modules.registration import weighted_procrustes
from pareconv.utils.registration import (
    compute_registration_error,
    evaluate_correspondences,
    evaluate_sparse_correspondences,
)

from config import get_benchmark_tag, make_cfg, make_experiment_parser


def make_parser():
    parser = make_experiment_parser(
        training=False,
        description='Evaluate saved PARE-Net correspondences.',
    )
    parser.add_argument(
        '--benchmark',
        default='3DMatch',
        choices=['3DMatch', '3DLoMatch'],
    )
    parser.add_argument(
        '--method',
        default='fhp',
        choices=['fhp', 'ransac', 'svd'],
        help='Rigid transformation estimator.',
    )
    parser.add_argument(
        '--num_corr',
        type=int,
        default=None,
        help='Use the top-N correspondences for RANSAC/SVD.',
    )
    parser.add_argument(
        '--estimator_seed',
        type=int,
        default=None,
        help=(
            'Random seed for the stochastic RANSAC estimator. '
            'Required for seeded RANSAC-control runs.'
        ),
    )
    parser.add_argument('--verbose', action='store_true')
    return parser


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _median(values):
    return float(np.median(values)) if values else 0.0


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


def _load_and_validate_manifest(features_root, cfg, benchmark):
    manifest_file = osp.join(features_root, 'manifest.json')
    if not osp.isfile(manifest_file):
        raise FileNotFoundError(
            f'Missing {manifest_file}. Re-run test.py with the revised code.'
        )
    with open(manifest_file, 'r', encoding='utf-8') as file:
        manifest = json.load(file)

    expected = {
        'code_version': getattr(cfg, 'code_version', None),
        'variant': cfg.variant,
        'seed': cfg.seed,
        're_feature_source': cfg.fine_matching.re_feature_source,
        'coarse_context': cfg.backbone.coarse_context,
        'fine_context': cfg.backbone.fine_context,
        'spsa_ratio': cfg.backbone.spsa_partial_ratio,
        'mspki_scales': list(cfg.backbone.mspki_neighbor_scales),
        'layer_scale_init': cfg.backbone.layer_scale_init,
        'num_hypotheses': cfg.fine_matching.num_hypotheses,
        'benchmark': benchmark,
        'rotation_mode': cfg.test.rotation_mode,
        'rotation_seed': cfg.test.rotation_seed,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f'Feature manifest/config mismatch: {mismatches}')

    if not manifest.get('completed', False):
        raise RuntimeError(
            'Feature extraction is marked incomplete. Re-run test.py; partial '
            'features must not be used for paper statistics.'
        )
    expected_pairs = int(manifest.get('expected_pairs', -1))
    saved_pairs = int(manifest.get('saved_pairs', -1))
    actual_pairs = len(glob.glob(osp.join(features_root, '*', '*.npz')))
    if expected_pairs < 0 or saved_pairs != expected_pairs or actual_pairs != expected_pairs:
        raise RuntimeError(
            'Feature count mismatch: '
            f'expected={expected_pairs}, manifest_saved={saved_pairs}, actual={actual_pairs}.'
        )
    return manifest


def _estimate_transform(args, cfg, data_dict, ref_points, src_points, scores):
    identity = np.eye(4, dtype=np.float32)

    if args.method == 'fhp':
        succeeded = bool(np.asarray(data_dict['registration_succeeded']).item())
        transform = np.asarray(data_dict['estimated_transform'])
        succeeded = succeeded and ref_points.shape[0] > 0 and _is_valid_rigid_transform(transform)
        return (transform if succeeded else identity), succeeded

    if ref_points.shape[0] < 3 or src_points.shape[0] < 3:
        return identity, False

    if args.method == 'ransac':
        from pareconv.utils.open3d import (
            registration_with_ransac_from_correspondences,
        )

        try:
            transform = registration_with_ransac_from_correspondences(
                src_points,
                ref_points,
                distance_threshold=cfg.ransac.distance_threshold,
                ransac_n=cfg.ransac.num_points,
                num_iterations=cfg.ransac.num_iterations,
            )
        except (RuntimeError, ValueError):
            return identity, False
        transform = np.asarray(transform)
        succeeded = _is_valid_rigid_transform(transform)
        return (transform if succeeded else identity), succeeded

    if args.method == 'svd':
        if (
            scores.shape[0] != src_points.shape[0]
            or not np.isfinite(scores).all()
            or not np.any(scores > 0)
        ):
            return identity, False
        centered = src_points - src_points.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 2:
            return identity, False
        with torch.no_grad():
            transform = weighted_procrustes(
                torch.from_numpy(src_points).float(),
                torch.from_numpy(ref_points).float(),
                torch.from_numpy(scores).float(),
                return_transform=True,
            ).cpu().numpy()
        succeeded = _is_valid_rigid_transform(transform)
        return (transform if succeeded else identity), succeeded

    raise ValueError(f'Unsupported registration method: {args.method}')


def _empty_fine_result():
    return {
        'inlier_ratio': 0.0,
        'overlap': 0.0,
        'residual': 0.0,
        'num_corr': 0,
    }



def _method_tag(args):
    if args.num_corr is None:
        tag = args.method
    else:
        tag = f'{args.method}_n{int(args.num_corr)}'
    if args.method == 'ransac' and args.estimator_seed is not None:
        tag += f'_s{int(args.estimator_seed)}'
    return tag


def _seed_ransac_estimator(args):
    if args.method != 'ransac':
        return
    if args.estimator_seed is None:
        raise ValueError(
            '--estimator_seed is required for paper RANSAC-control runs so '
            'that repeated estimator randomness is explicit and reproducible.'
        )

    seed = int(args.estimator_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    import open3d as o3d

    random_api = getattr(o3d.utility, 'random', None)
    seed_fn = getattr(random_api, 'seed', None) if random_api is not None else None
    if seed_fn is None:
        raise RuntimeError(
            'This Open3D build does not expose o3d.utility.random.seed(). '
            'Do not claim seeded RANSAC reproducibility with this build.'
        )
    seed_fn(seed)


def _read_hypothesis_diagnostics(data_dict):
    keys = (
        'num_hypotheses',
        'num_encoder_hypotheses',
        'num_decoder_hypotheses',
        'num_successful_hypotheses',
        'num_successful_encoder_hypotheses',
        'num_successful_decoder_hypotheses',
        'oracle_success',
        'encoder_oracle_success',
        'decoder_oracle_success',
        'selected_pose_success',
        'raw_selected_pose_success',
        'shortlist_oracle_success',
        'refined_shortlist_oracle_success',
        'num_shortlisted_hypotheses',
        'num_successful_shortlisted_hypotheses',
        'num_successful_refined_shortlisted_hypotheses',
        'selected_hypothesis_index',
        'selected_hypothesis_source',
        'selected_pre_refine_consensus',
        'selected_pre_refine_inlier_count',
        'best_encoder_consensus',
        'best_decoder_consensus',
        'raw_selected_hypothesis_index',
        'raw_selected_hypothesis_source',
        'raw_selected_pre_refine_consensus',
        'raw_selected_pre_refine_inlier_count',
        'raw_selected_post_refine_weighted_consensus',
        'selected_post_refine_weighted_consensus',
        'best_encoder_post_refine_weighted_consensus',
        'best_decoder_post_refine_weighted_consensus',
        'selection_changed_from_raw_argmax',
        'reranking_applied',
    )
    if any(key not in data_dict.files for key in keys):
        return None
    return {
        key: np.asarray(data_dict[key]).item()
        for key in keys
    }


def _summarize_hypothesis_diagnostics(pair_rows, cfg):
    rows = [row for row in pair_rows if row.get('oracle_success') is not None]
    if not rows:
        return None

    num_pairs = len(rows)
    encoder_only = sum(
        bool(row['encoder_oracle_success'])
        and not bool(row['decoder_oracle_success'])
        for row in rows
    )
    decoder_only = sum(
        bool(row['decoder_oracle_success'])
        and not bool(row['encoder_oracle_success'])
        for row in rows
    )
    selected_encoder = sum(
        int(row['selected_hypothesis_source']) == 0 for row in rows
    )
    selected_decoder = sum(
        int(row['selected_hypothesis_source']) == 1 for row in rows
    )
    raw_selected_encoder = sum(
        int(row['raw_selected_hypothesis_source']) == 0 for row in rows
    )
    raw_selected_decoder = sum(
        int(row['raw_selected_hypothesis_source']) == 1 for row in rows
    )
    oracle_pairs = sum(bool(row['oracle_success']) for row in rows)

    def finite_mean(key):
        values = [
            float(row[key])
            for row in rows
            if np.isfinite(float(row[key]))
        ]
        return _mean(values)

    return {
        'success_thresholds': {
            'rre_degrees': cfg.eval.rre_threshold,
            'rte_meters': cfg.eval.rte_threshold,
        },
        'num_pairs': num_pairs,
        'encoder_oracle_success_rate': _mean(
            [float(row['encoder_oracle_success']) for row in rows]
        ),
        'decoder_oracle_success_rate': _mean(
            [float(row['decoder_oracle_success']) for row in rows]
        ),
        'union_oracle_success_rate': _mean(
            [float(row['oracle_success']) for row in rows]
        ),
        'selected_pose_success_rate': _mean(
            [float(row['selected_pose_success']) for row in rows]
        ),
        'raw_selected_pose_success_rate': _mean(
            [float(row['raw_selected_pose_success']) for row in rows]
        ),
        'reranking_success_rate_gain': _mean(
            [
                float(row['selected_pose_success'])
                - float(row['raw_selected_pose_success'])
                for row in rows
            ]
        ),
        'shortlist_oracle_success_rate': _mean(
            [float(row['shortlist_oracle_success']) for row in rows]
        ),
        'refined_shortlist_oracle_success_rate': _mean(
            [
                float(row['refined_shortlist_oracle_success'])
                for row in rows
            ]
        ),
        'shortlist_oracle_retention_rate': (
            sum(
                bool(row['oracle_success'])
                and bool(row['shortlist_oracle_success'])
                for row in rows
            )
            / oracle_pairs
            if oracle_pairs
            else 0.0
        ),
        'mean_hypotheses': _mean(
            [row['num_hypotheses'] for row in rows]
        ),
        'mean_shortlisted_hypotheses': _mean(
            [row['num_shortlisted_hypotheses'] for row in rows]
        ),
        'mean_successful_shortlisted_hypotheses': _mean(
            [row['num_successful_shortlisted_hypotheses'] for row in rows]
        ),
        'mean_successful_refined_shortlisted_hypotheses': _mean(
            [
                row['num_successful_refined_shortlisted_hypotheses']
                for row in rows
            ]
        ),
        'mean_encoder_hypotheses': _mean(
            [row['num_encoder_hypotheses'] for row in rows]
        ),
        'mean_decoder_hypotheses': _mean(
            [row['num_decoder_hypotheses'] for row in rows]
        ),
        'mean_successful_encoder_hypotheses': _mean(
            [row['num_successful_encoder_hypotheses'] for row in rows]
        ),
        'mean_successful_decoder_hypotheses': _mean(
            [row['num_successful_decoder_hypotheses'] for row in rows]
        ),
        'encoder_only_success_pairs': int(encoder_only),
        'decoder_only_success_pairs': int(decoder_only),
        'selected_encoder_pairs': int(selected_encoder),
        'selected_decoder_pairs': int(selected_decoder),
        'raw_selected_encoder_pairs': int(raw_selected_encoder),
        'raw_selected_decoder_pairs': int(raw_selected_decoder),
        'selection_changed_pairs': int(
            sum(bool(row['selection_changed_from_raw_argmax']) for row in rows)
        ),
        'reranking_rescued_pairs': int(
            sum(
                not bool(row['raw_selected_pose_success'])
                and bool(row['selected_pose_success'])
                for row in rows
            )
        ),
        'reranking_harmed_pairs': int(
            sum(
                bool(row['raw_selected_pose_success'])
                and not bool(row['selected_pose_success'])
                for row in rows
            )
        ),
        'mean_selected_pre_refine_consensus': finite_mean(
            'selected_pre_refine_consensus'
        ),
        'mean_best_encoder_consensus': finite_mean('best_encoder_consensus'),
        'mean_best_decoder_consensus': finite_mean('best_decoder_consensus'),
        'mean_raw_selected_post_refine_weighted_consensus': finite_mean(
            'raw_selected_post_refine_weighted_consensus'
        ),
        'mean_selected_post_refine_weighted_consensus': finite_mean(
            'selected_post_refine_weighted_consensus'
        ),
        'mean_best_encoder_post_refine_weighted_consensus': finite_mean(
            'best_encoder_post_refine_weighted_consensus'
        ),
        'mean_best_decoder_post_refine_weighted_consensus': finite_mean(
            'best_decoder_post_refine_weighted_consensus'
        ),
        'encoder_consensus_win_pairs': int(
            sum(
                np.isfinite(float(row['best_encoder_consensus']))
                and np.isfinite(float(row['best_decoder_consensus']))
                and float(row['best_encoder_consensus'])
                > float(row['best_decoder_consensus'])
                for row in rows
            )
        ),
        'decoder_consensus_win_pairs': int(
            sum(
                np.isfinite(float(row['best_encoder_consensus']))
                and np.isfinite(float(row['best_decoder_consensus']))
                and float(row['best_decoder_consensus'])
                > float(row['best_encoder_consensus'])
                for row in rows
            )
        ),
        'generation_failure_pairs': int(
            sum(not bool(row['oracle_success']) for row in rows)
        ),
        'selection_failure_pairs': int(
            sum(
                bool(row['oracle_success'])
                and not bool(row['selected_pose_success'])
                for row in rows
            )
        ),
        'shortlist_miss_pairs': int(
            sum(
                bool(row['oracle_success'])
                and not bool(row['shortlist_oracle_success'])
                for row in rows
            )
        ),
        'post_refine_selection_failure_pairs': int(
            sum(
                bool(row['refined_shortlist_oracle_success'])
                and not bool(row['selected_pose_success'])
                for row in rows
            )
        ),
    }


def eval_one_epoch(args, cfg, logger):
    if args.method == 'fhp' and args.num_corr is not None:
        raise ValueError(
            '--num_corr cannot be used with --method fhp because the saved FHP '
            'transform was already estimated during test.py. Use ransac or svd.'
        )
    if args.num_corr is not None and args.num_corr < 1:
        raise ValueError('--num_corr must be positive.')
    if args.method == 'ransac':
        try:
            import open3d
        except ImportError as exc:
            raise RuntimeError(
                'Open3D is required for --method ransac. Install a compatible '
                'Open3D build or use --method fhp/--method svd.'
            ) from exc
        _seed_ransac_estimator(args)

    benchmark_tag = get_benchmark_tag(cfg, args.benchmark)
    method_tag = _method_tag(args)
    features_root = osp.join(cfg.feature_dir, benchmark_tag)
    manifest = _load_and_validate_manifest(features_root, cfg, args.benchmark)
    logger.info(f'Feature manifest: {json.dumps(manifest, indent=2)}')

    scene_roots = sorted(
        path
        for path in glob.glob(osp.join(features_root, '*'))
        if osp.isdir(path)
    )
    if not scene_roots:
        raise FileNotFoundError(f'No scene features found under {features_root}.')

    scene_results = {}
    pair_rows = []
    aggregate = {
        'PIR': [],
        'FMR': [],
        'IR': [],
        'OV': [],
        'RR': [],
        'mean_RRE': [],
        'mean_RTE': [],
        'median_RRE': [],
        'median_RTE': [],
    }

    for scene_root in scene_roots:
        scene_name = osp.basename(scene_root)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)

        gt_indices = gt_logs = gt_infos = None
        if cfg.test.rotation_mode == 'none':
            gt_root = osp.join(
                cfg.data.metadata_root,
                'benchmarks',
                args.benchmark,
                scene_name,
            )
            gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(
                gt_root, num_fragments
            )
            num_gt_pairs = int((gt_indices != -1).sum())
        else:
            num_gt_pairs = 0

        file_names = sorted(
            glob.glob(osp.join(scene_root, '*.npz')),
            key=lambda path: [
                int(item)
                for item in osp.basename(path).split('.')[0].split('_')
            ],
        )
        if not file_names:
            raise RuntimeError(f'No .npz files found for scene {scene_name}.')

        coarse_precisions = []
        fmr_flags = []
        inlier_ratios = []
        overlaps = []
        accepted_flags = []
        accepted_rres = []
        accepted_rtes = []
        estimated_transforms = []
        failed_pairs = []
        num_pred_gt_pairs = 0
        failed_estimations = 0

        for file_name in file_names:
            ref_frame, src_frame = [
                int(item)
                for item in osp.basename(file_name).split('.')[0].split('_')
            ]
            with np.load(file_name) as data_dict:
                ref_points_c = data_dict['ref_points_c']
                src_points_c = data_dict['src_points_c']
                ref_node_corr_indices = data_dict['ref_node_corr_indices']
                src_node_corr_indices = data_dict['src_node_corr_indices']
                gt_node_corr_indices = data_dict['gt_node_corr_indices']
                saved_transform = data_dict['transform']
                pcd_overlap = float(data_dict['overlap'])

                ref_corr_points = data_dict['ref_corr_points']
                src_corr_points = data_dict['src_corr_points']
                corr_scores = data_dict['corr_scores']
                hypothesis_diagnostics = _read_hypothesis_diagnostics(data_dict)

                if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
                    selected = np.argsort(-corr_scores)[: args.num_corr]
                    ref_corr_points = ref_corr_points[selected]
                    src_corr_points = src_corr_points[selected]
                    corr_scores = corr_scores[selected]

                coarse_result = evaluate_sparse_correspondences(
                    ref_points_c,
                    src_points_c,
                    ref_node_corr_indices,
                    src_node_corr_indices,
                    gt_node_corr_indices,
                )
                coarse_precision = float(coarse_result['precision'])
                coarse_precisions.append(coarse_precision)

                if ref_corr_points.shape[0] == 0:
                    fine_result = _empty_fine_result()
                else:
                    fine_result = evaluate_correspondences(
                        ref_corr_points,
                        src_corr_points,
                        saved_transform,
                        positive_radius=cfg.eval.acceptance_radius,
                    )
                inlier_ratio = float(fine_result['inlier_ratio'])
                overlap = float(fine_result['overlap'])
                inlier_ratios.append(inlier_ratio)
                overlaps.append(overlap)
                fmr_flags.append(
                    float(inlier_ratio >= cfg.eval.inlier_ratio_threshold)
                )

                estimated_transform, estimation_succeeded = _estimate_transform(
                    args,
                    cfg,
                    data_dict,
                    ref_corr_points,
                    src_corr_points,
                    corr_scores,
                )
                if not estimation_succeeded:
                    failed_estimations += 1
                if estimation_succeeded:
                    estimated_transforms.append(
                        {
                            'test_pair': [ref_frame, src_frame],
                            'num_fragments': num_fragments,
                            'transform': estimated_transform,
                        }
                    )
                else:
                    failed_pairs.append([ref_frame, src_frame])

                accepted = False
                registration_error = None
                if cfg.test.rotation_mode == 'none':
                    gt_index = int(gt_indices[ref_frame, src_frame])
                    if gt_index != -1:
                        if estimation_succeeded:
                            num_pred_gt_pairs += 1
                            gt_transform = gt_logs[gt_index]['transform']
                            covariance = gt_infos[gt_index]['covariance']
                            registration_error = compute_transform_error(
                                gt_transform, covariance, estimated_transform
                            )
                            accepted = bool(
                                np.isfinite(registration_error)
                                and registration_error <= cfg.eval.rmse_threshold**2
                            )
                            if accepted:
                                rre, rte = compute_registration_error(
                                    gt_transform, estimated_transform
                                )
                                accepted_rres.append(float(rre))
                                accepted_rtes.append(float(rte))
                        accepted_flags.append(float(accepted))
                else:
                    if estimation_succeeded:
                        rre, rte = compute_registration_error(
                            saved_transform, estimated_transform
                        )
                        accepted = bool(
                            np.isfinite(rre)
                            and np.isfinite(rte)
                            and rre <= cfg.eval.rre_threshold
                            and rte <= cfg.eval.rte_threshold
                        )
                        if accepted:
                            accepted_rres.append(float(rre))
                            accepted_rtes.append(float(rte))
                    accepted_flags.append(float(accepted))

                pair_row = {
                    'scene': scene_name,
                    'ref_frame': ref_frame,
                    'src_frame': src_frame,
                    'overlap': pcd_overlap,
                    'coarse_precision': coarse_precision,
                    'inlier_ratio': inlier_ratio,
                    'estimation_succeeded': int(estimation_succeeded),
                    'registration_accepted': int(accepted),
                }
                if hypothesis_diagnostics is not None:
                    pair_row.update(hypothesis_diagnostics)
                else:
                    pair_row.update(
                        {
                            'num_hypotheses': None,
                            'num_encoder_hypotheses': None,
                            'num_decoder_hypotheses': None,
                            'num_successful_hypotheses': None,
                            'num_successful_encoder_hypotheses': None,
                            'num_successful_decoder_hypotheses': None,
                            'oracle_success': None,
                            'encoder_oracle_success': None,
                            'decoder_oracle_success': None,
                            'selected_pose_success': None,
                            'raw_selected_pose_success': None,
                            'shortlist_oracle_success': None,
                            'refined_shortlist_oracle_success': None,
                            'num_shortlisted_hypotheses': None,
                            'num_successful_shortlisted_hypotheses': None,
                            'num_successful_refined_shortlisted_hypotheses': None,
                            'selected_hypothesis_index': None,
                            'selected_hypothesis_source': None,
                            'selected_pre_refine_consensus': None,
                            'selected_pre_refine_inlier_count': None,
                            'best_encoder_consensus': None,
                            'best_decoder_consensus': None,
                            'raw_selected_hypothesis_index': None,
                            'raw_selected_hypothesis_source': None,
                            'raw_selected_pre_refine_consensus': None,
                            'raw_selected_pre_refine_inlier_count': None,
                            'raw_selected_post_refine_weighted_consensus': None,
                            'selected_post_refine_weighted_consensus': None,
                            'best_encoder_post_refine_weighted_consensus': None,
                            'best_decoder_post_refine_weighted_consensus': None,
                            'selection_changed_from_raw_argmax': None,
                            'reranking_applied': None,
                        }
                    )
                selected_source = pair_row['selected_hypothesis_source']
                pair_row['selected_hypothesis_source_name'] = (
                    'encoder'
                    if selected_source == 0
                    else 'decoder'
                    if selected_source == 1
                    else 'none'
                )
                raw_selected_source = pair_row[
                    'raw_selected_hypothesis_source'
                ]
                pair_row['raw_selected_hypothesis_source_name'] = (
                    'encoder'
                    if raw_selected_source == 0
                    else 'decoder'
                    if raw_selected_source == 1
                    else 'none'
                )
                pair_rows.append(pair_row)

                if args.verbose:
                    message = (
                        f'{scene_abbr}, id0={ref_frame}, id1={src_frame}, '
                        f'OV={pcd_overlap:.3f}, c_PIR={coarse_precision:.3f}, '
                        f'f_IR={inlier_ratio:.3f}, f_OV={overlap:.3f}, '
                        f'f_RS={float(fine_result["residual"]):.3f}, '
                        f'f_NU={int(fine_result["num_corr"])}, '
                        f'est_success={int(estimation_succeeded)}'
                    )
                    if registration_error is not None:
                        message += f', r_RMSE={np.sqrt(registration_error):.3f}'
                    message += f', accepted={int(accepted)}'
                    logger.info(message)

        est_log = osp.join(
            cfg.registration_dir,
            benchmark_tag,
            scene_name,
            f'{method_tag}.log',
        )
        write_log_file(est_log, estimated_transforms)

        if cfg.test.rotation_mode == 'none':
            rr = (
                float(sum(accepted_flags)) / num_gt_pairs
                if num_gt_pairs > 0
                else 0.0
            )
            precision = (
                float(sum(accepted_flags)) / num_pred_gt_pairs
                if num_pred_gt_pairs > 0
                else 0.0
            )
        else:
            rr = _mean(accepted_flags)
            precision = rr
            num_gt_pairs = len(file_names)
            num_pred_gt_pairs = len(file_names)

        result = {
            'PIR': _mean(coarse_precisions),
            'FMR': _mean(fmr_flags),
            'IR': _mean(inlier_ratios),
            'OV': _mean(overlaps),
            'RR': rr,
            'registration_precision': precision,
            'mean_RRE': _mean(accepted_rres),
            'mean_RTE': _mean(accepted_rtes),
            'median_RRE': _median(accepted_rres),
            'median_RTE': _median(accepted_rtes),
            'num_gt_pairs': num_gt_pairs,
            'num_pred_gt_pairs': num_pred_gt_pairs,
            'num_accepted': int(sum(accepted_flags)),
            'num_failed_estimations': int(failed_estimations),
            'failed_pairs': failed_pairs,
        }
        scene_results[scene_abbr] = result
        for key in aggregate:
            aggregate[key].append(result[key])

        logger.info(
            f'{scene_abbr}: PIR={result["PIR"]:.3f}, '
            f'FMR={result["FMR"]:.3f}, IR={result["IR"]:.3f}, '
            f'RR={result["RR"]:.3f}, failures={failed_estimations}, '
            f'mean_RRE={result["mean_RRE"]:.3f}, '
            f'mean_RTE={result["mean_RTE"]:.3f}'
        )

    overall = {key: _mean(values) for key, values in aggregate.items()}
    logger.critical(
        f'Overall: PIR={overall["PIR"]:.3f}, FMR={overall["FMR"]:.3f}, '
        f'IR={overall["IR"]:.3f}, OV={overall["OV"]:.3f}, '
        f'RR={overall["RR"]:.3f}, mean_RRE={overall["mean_RRE"]:.3f}, '
        f'mean_RTE={overall["mean_RTE"]:.3f}, '
        f'median_RRE={overall["median_RRE"]:.3f}, '
        f'median_RTE={overall["median_RTE"]:.3f}'
    )

    hypothesis_summary = _summarize_hypothesis_diagnostics(pair_rows, cfg)
    pair_file = osp.join(
        cfg.registration_dir,
        benchmark_tag,
        f'pairs_{method_tag}.csv',
    )
    Path(pair_file).parent.mkdir(parents=True, exist_ok=True)
    with open(pair_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        'code_version': getattr(cfg, 'code_version', None),
        'variant': cfg.variant,
        'seed': cfg.seed,
        're_feature_source': cfg.fine_matching.re_feature_source,
        'num_hypotheses': cfg.fine_matching.num_hypotheses,
        'coarse_context': cfg.backbone.coarse_context,
        'fine_context': cfg.backbone.fine_context,
        'spsa_ratio': cfg.backbone.spsa_partial_ratio,
        'mspki_scales': list(cfg.backbone.mspki_neighbor_scales),
        'layer_scale_init': cfg.backbone.layer_scale_init,
        'benchmark': args.benchmark,
        'benchmark_tag': benchmark_tag,
        'rotation_mode': cfg.test.rotation_mode,
        'rotation_seed': cfg.test.rotation_seed,
        'registration_protocol': (
            'official_covariance_rr'
            if cfg.test.rotation_mode == 'none'
            else 'rre_rte_success_rate'
        ),
        'method': args.method,
        'method_tag': method_tag,
        'num_corr': args.num_corr,
        'estimator_seed': args.estimator_seed,
        'feature_manifest': manifest,
        'hypothesis_diagnostics': hypothesis_summary,
        'pair_diagnostics_file': pair_file,
        'overall': overall,
        'scenes': scene_results,
    }
    summary_file = osp.join(
        cfg.registration_dir,
        benchmark_tag,
        f'summary_{method_tag}.json',
    )
    Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    logger.critical(f'Summary saved to {summary_file}')
    return summary


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg(args)
    log_file = osp.join(
        cfg.log_dir,
        f'eval-{time.strftime("%Y%m%d-%H%M%S")}.log',
    )
    logger = Logger(log_file=log_file)
    logger.info('Command executed: ' + ' '.join(sys.argv))
    logger.info('Configs:\n' + json.dumps(cfg, indent=4))
    eval_one_epoch(args, cfg, logger)


if __name__ == '__main__':
    main()
