
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def registration_error(gt, est):
    gt = np.asarray(gt, dtype=float)
    est = np.asarray(est, dtype=float)
    r = gt[:3, :3].T @ est[:3, :3]
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    rre = np.degrees(np.arccos(cos))
    rte = np.linalg.norm(gt[:3, 3] - est[:3, 3])
    return float(rre), float(rte)


def parser():
    p = argparse.ArgumentParser(
        description='Select and package qualitative candidates for 3DLoMatch or KITTI.'
    )
    p.add_argument('--task', required=True, choices=['indoor', 'kitti'])
    p.add_argument('--baseline_dir', required=True,
                   help='.../baseline/re_official/seed_7351')
    p.add_argument('--mspki_dir', required=True,
                   help='.../mspki/re_official/seed_7351')
    p.add_argument('--benchmark', default='3DLoMatch')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--topn_each', type=int, default=12)
    p.add_argument('--min_ir_gain_pp', type=float, default=5.0)
    return p


def indoor_paths(root: Path, benchmark: str):
    return root / 'registration' / benchmark / 'pairs_fhp.csv', root / 'features' / benchmark


def kitti_paths(root: Path):
    return root / 'registration' / 'pairs_fhp.csv', root / 'features'


def normalize_indoor(b, m):
    keys = ['scene', 'ref_frame', 'src_frame']
    for df in (b, m):
        df['ref_frame'] = df['ref_frame'].astype(int)
        df['src_frame'] = df['src_frame'].astype(int)
    x = b.merge(m, on=keys, suffixes=('_baseline', '_mspki'), validate='one_to_one')
    x['IR_baseline_pct'] = 100.0 * x['inlier_ratio_baseline'].astype(float)
    x['IR_mspki_pct'] = 100.0 * x['inlier_ratio_mspki'].astype(float)
    x['delta_IR_pp'] = x['IR_mspki_pct'] - x['IR_baseline_pct']
    x['baseline_success'] = x['registration_accepted_baseline'].astype(int)
    x['mspki_success'] = x['registration_accepted_mspki'].astype(int)
    x['rescue'] = ((x['baseline_success'] == 0) & (x['mspki_success'] == 1)).astype(int)
    x['both_success'] = ((x['baseline_success'] == 1) & (x['mspki_success'] == 1)).astype(int)
    return x


def normalize_kitti(b, m):
    keys = ['seq', 'src_frame', 'ref_frame']
    for df in (b, m):
        for c in keys:
            df[c] = df[c].astype(int)
    x = b.merge(m, on=keys, suffixes=('_baseline', '_mspki'), validate='one_to_one')
    x['IR_baseline_pct'] = 100.0 * x['IR_baseline'].astype(float)
    x['IR_mspki_pct'] = 100.0 * x['IR_mspki'].astype(float)
    x['delta_IR_pp'] = x['IR_mspki_pct'] - x['IR_baseline_pct']
    x['baseline_success'] = x['TR_baseline'].astype(int)
    x['mspki_success'] = x['TR_mspki'].astype(int)
    x['rescue'] = ((x['baseline_success'] == 0) & (x['mspki_success'] == 1)).astype(int)
    x['both_success'] = ((x['baseline_success'] == 1) & (x['mspki_success'] == 1)).astype(int)
    return x


def indoor_npz(feature_dir: Path, row):
    return feature_dir / str(row['scene']) / f"{int(row['ref_frame'])}_{int(row['src_frame'])}.npz"


def kitti_npz(feature_dir: Path, row):
    return feature_dir / f"{int(row['seq'])}_{int(row['src_frame'])}_{int(row['ref_frame'])}.npz"


def enrich_pose_errors(x, task, fb, fm):
    vals = []
    for _, row in x.iterrows():
        pb = indoor_npz(fb, row) if task == 'indoor' else kitti_npz(fb, row)
        pm = indoor_npz(fm, row) if task == 'indoor' else kitti_npz(fm, row)
        if not pb.is_file() or not pm.is_file():
            vals.append((np.nan, np.nan, np.nan, np.nan, False))
            continue
        zb = np.load(pb)
        zm = np.load(pm)
        gt = np.asarray(zb['transform'], dtype=float)
        rreb, rteb = registration_error(gt, np.asarray(zb['estimated_transform'], dtype=float))
        rrem, rtem = registration_error(gt, np.asarray(zm['estimated_transform'], dtype=float))
        vals.append((rreb, rteb, rrem, rtem, True))
    x = x.copy()
    x[['RRE_baseline_deg','RTE_baseline_m','RRE_mspki_deg','RTE_mspki_m','npz_available']] = pd.DataFrame(vals, index=x.index)
    return x


def main():
    args = parser().parse_args()
    bdir, mdir = Path(args.baseline_dir), Path(args.mspki_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.task == 'indoor':
        bcsv, fb = indoor_paths(bdir, args.benchmark)
        mcsv, fm = indoor_paths(mdir, args.benchmark)
    else:
        bcsv, fb = kitti_paths(bdir)
        mcsv, fm = kitti_paths(mdir)

    for p in (bcsv, mcsv):
        if not p.is_file():
            raise FileNotFoundError(p)

    b, m = pd.read_csv(bcsv), pd.read_csv(mcsv)
    x = normalize_indoor(b, m) if args.task == 'indoor' else normalize_kitti(b, m)
    x = enrich_pose_errors(x, args.task, fb, fm)


    rescue = x[(x['rescue'] == 1) & x['npz_available']].copy()
    rescue = rescue.sort_values(['delta_IR_pp', 'IR_mspki_pct'], ascending=[False, False])

    both = x[(x['both_success'] == 1) & x['npz_available']].copy()
    both = both[both['delta_IR_pp'] >= float(args.min_ir_gain_pp)]
    both = both.sort_values(['delta_IR_pp', 'IR_mspki_pct'], ascending=[False, False])

    rescue.head(args.topn_each).to_csv(out / f'{args.task}_rescue_candidates.csv', index=False)
    both.head(args.topn_each).to_csv(out / f'{args.task}_both_success_ir_gain_candidates.csv', index=False)
    x.to_csv(out / f'{args.task}_all_paired_metrics.csv', index=False)

    selected = pd.concat([
        rescue.head(args.topn_each).assign(candidate_type='rescue'),
        both.head(args.topn_each).assign(candidate_type='both_success_ir_gain'),
    ], ignore_index=True)

    pkg = out / 'candidate_npz'
    pkg.mkdir(exist_ok=True)
    copied = []
    seen = set()
    for _, row in selected.iterrows():
        if args.task == 'indoor':
            ident = f"{row['scene']}__ref{int(row['ref_frame'])}_src{int(row['src_frame'])}"
            pb, pm = indoor_npz(fb, row), indoor_npz(fm, row)
        else:
            ident = f"seq{int(row['seq']):02d}__ref{int(row['ref_frame'])}_src{int(row['src_frame'])}"
            pb, pm = kitti_npz(fb, row), kitti_npz(fm, row)
        if ident in seen:
            continue
        seen.add(ident)
        dst_b = pkg / f'{ident}__baseline.npz'
        dst_m = pkg / f'{ident}__mspki.npz'
        shutil.copy2(pb, dst_b)
        shutil.copy2(pm, dst_m)
        copied.append({'id': ident, 'baseline': str(dst_b), 'mspki': str(dst_m)})

    (out / f'{args.task}_package_manifest.json').write_text(
        json.dumps({'task': args.task, 'copied': copied}, indent=2), encoding='utf-8'
    )
    print('All pairs:', len(x))
    print('Rescues:', len(rescue))
    print('Both-success with IR gain >=', args.min_ir_gain_pp, 'pp:', len(both))
    print('Copied candidate pairs:', len(copied))
    print('Saved:', out)


if __name__ == '__main__':
    main()
