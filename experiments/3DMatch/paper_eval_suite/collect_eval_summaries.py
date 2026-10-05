from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Collect eval.py summary JSON files into one CSV.')
    parser.add_argument('--search_root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows = []
    for path in Path(args.search_root).rglob('summary_*.json'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        overall = data.get('overall', {})
        manifest = data.get('feature_manifest') or {}
        row = {
            'summary_file': str(path),
            'variant': data.get('variant', manifest.get('variant')),
            'seed': data.get('seed', manifest.get('seed')),
            'benchmark': data.get('benchmark', manifest.get('benchmark')),
            're_feature_source': data.get('re_feature_source', manifest.get('re_feature_source')),
            'method': data.get('method'),
            'rotation_mode': data.get('rotation_mode', manifest.get('rotation_mode', 'none')),
            'rotation_seed': data.get('rotation_seed', manifest.get('rotation_seed')),
            'stress_type': manifest.get('stress_type', 'none'),
            'noise_sigma_m': manifest.get('noise_sigma_m', 0.0),
            'density_ratio': manifest.get('density_ratio', 1.0),
            'density_target': manifest.get('density_target', ''),
            'stress_seed': manifest.get('stress_seed', ''),
            'PIR': overall.get('PIR'),
            'FMR': overall.get('FMR'),
            'IR': overall.get('IR'),
            'RR': overall.get('RR'),
            'mean_RRE': overall.get('mean_RRE'),
            'mean_RTE': overall.get('mean_RTE'),
            'median_RRE': overall.get('median_RRE'),
            'median_RTE': overall.get('median_RTE'),
        }
        rows.append(row)

    rows.sort(key=lambda r: (
        str(r['stress_type']), str(r['benchmark']), str(r['variant']),
        int(r['seed']) if str(r['seed']).isdigit() else -1,
        float(r['noise_sigma_m'] or 0), float(r['density_ratio'] or 1),
        int(r['rotation_seed']) if str(r['rotation_seed']).isdigit() else -1,
    ))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ['summary_file']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Collected {len(rows)} summaries -> {args.output}')


if __name__ == '__main__':
    main()
