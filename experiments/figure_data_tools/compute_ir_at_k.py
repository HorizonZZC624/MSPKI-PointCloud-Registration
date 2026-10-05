
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description='Compute score-ranked IR@K from saved correspondence NPZ files.')
    p.add_argument('--feature_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--benchmark', required=True, choices=['3DMatch','3DLoMatch'])
    p.add_argument('--ks', nargs='*', type=int, default=[50,100,250,500])
    p.add_argument('--acceptance_radius', type=float, default=0.1)
    return p


def transform(T, pts):
    T = np.asarray(T, dtype=float)
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3,:3].T + T[:3,3]


def main():
    args = parser().parse_args()
    root = Path(args.feature_dir)
    files = sorted(p for p in root.rglob('*.npz') if p.is_file())
    if not files:
        raise FileNotFoundError(f'No NPZ files under {root}')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    budgets = list(args.ks) + ['all']
    rows = []
    for fn in files:
        z = np.load(fn)
        needed = ['ref_corr_points','src_corr_points','corr_scores','transform']
        missing = [k for k in needed if k not in z.files]
        if missing:
            raise ValueError(f'{fn} missing {missing}')
        ref = np.asarray(z['ref_corr_points'], dtype=float)
        src = np.asarray(z['src_corr_points'], dtype=float)
        scores = np.asarray(z['corr_scores'], dtype=float).reshape(-1)
        T = np.asarray(z['transform'], dtype=float)
        n = min(len(ref), len(src), len(scores))
        ref, src, scores = ref[:n], src[:n], scores[:n]
        residual = np.linalg.norm(ref - transform(T, src), axis=1) if n else np.empty((0,))
        order = np.argsort(-scores) if n else np.empty((0,), dtype=int)
        for k in budgets:
            if k == 'all':
                idx = order
                k_label = 'all'
            else:
                idx = order[:min(int(k), n)]
                k_label = str(int(k))
            ir = float(np.mean(residual[idx] < args.acceptance_radius)) if len(idx) else 0.0
            rows.append({
                'variant': args.variant,
                'seed': args.seed,
                'benchmark': args.benchmark,
                'pair_file': str(fn.relative_to(root)),
                'K': k_label,
                'num_available_corr': int(n),
                'num_used_corr': int(len(idx)),
                'IR': ir,
            })

    per_pair = out / 'ir_at_k_per_pair.csv'
    with per_pair.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = []
    for k in budgets:
        kl = 'all' if k == 'all' else str(int(k))
        vals = np.asarray([r['IR'] for r in rows if r['K'] == kl], dtype=float)
        used = np.asarray([r['num_used_corr'] for r in rows if r['K'] == kl], dtype=float)
        summary.append({
            'variant': args.variant,
            'seed': args.seed,
            'benchmark': args.benchmark,
            'K': kl,
            'num_pairs': int(len(vals)),
            'IR_mean': float(vals.mean()),
            'IR_std_across_pairs': float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
            'mean_used_corr': float(used.mean()) if len(used) else 0.0,
        })
    summ_csv = out / 'ir_at_k_summary.csv'
    with summ_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    (out / 'ir_at_k_protocol.json').write_text(json.dumps({
        'selection': 'descending corr_scores; top-K',
        'acceptance_radius_m': args.acceptance_radius,
        'budgets': [str(k) for k in budgets],
        'feature_dir': str(root.resolve()),
    }, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
