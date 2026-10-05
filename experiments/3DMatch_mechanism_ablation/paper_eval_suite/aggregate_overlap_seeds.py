from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def main():
    parser = argparse.ArgumentParser(description='Aggregate overlap-bin CSVs across model seeds.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--metrics', nargs='+',
        default=['PIR_macro_scene', 'FMR_macro_scene', 'IR_macro_scene', 'RR_macro_scene'],
    )
    args = parser.parse_args()

    grouped = defaultdict(lambda: defaultdict(list))
    meta = {}
    for file_name in args.inputs:
        with open(file_name, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (row['variant'], row['benchmark'], row['bin'])
                meta[key] = {
                    'variant': row['variant'], 'benchmark': row['benchmark'], 'bin': row['bin'],
                    'bin_low': row['bin_low'], 'bin_high': row['bin_high']
                }
                for metric in args.metrics:
                    value = _float(row.get(metric))
                    if np.isfinite(value):
                        grouped[key][metric].append(value)

    rows = []
    for key in sorted(meta, key=lambda x: (x[0], x[1], float(meta[x]['bin_low']))):
        row = dict(meta[key])
        n = 0
        for metric in args.metrics:
            values = grouped[key][metric]
            n = max(n, len(values))
            row[f'{metric}_mean'] = float(np.mean(values)) if values else float('nan')
            row[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else float('nan')
        row['num_seeds'] = n
        rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(rows[0].keys()) if rows else ['variant']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
