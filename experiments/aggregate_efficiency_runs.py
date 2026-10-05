from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--save_json', default=None)
    args = parser.parse_args()

    root = Path(args.root)
    result = {}

    for benchmark in ['3DMatch', '3DLoMatch']:
        result[benchmark] = {}
        for variant in ['baseline', 'mspki']:
            files = sorted(glob.glob(str(root / f'{variant}_{benchmark}_rep*.json')))
            if len(files) != args.repeats:
                raise RuntimeError(
                    f'{variant} {benchmark}: expected {args.repeats} files, found {len(files)}: {files}'
                )

            rows = [json.load(open(f, encoding='utf-8')) for f in files]
            latency = np.asarray([r['mean_ms'] for r in rows], dtype=float)
            median = np.asarray([r['median_ms'] for r in rows], dtype=float)
            p95 = np.asarray([r['p95_ms'] for r in rows], dtype=float)
            peak_alloc = np.asarray([r['peak_gpu_memory_mib'] for r in rows], dtype=float)
            peak_reserved = np.asarray([r.get('peak_gpu_reserved_mib', np.nan) for r in rows], dtype=float)

            item = {
                'parameters': int(rows[0]['parameters']),
                'trainable_parameters': int(rows[0]['trainable_parameters']),
                'process_repeats': len(rows),
                'latency_mean_of_run_means_ms': float(latency.mean()),
                'latency_sd_of_run_means_ms': float(latency.std(ddof=1)),
                'latency_run_means_ms': latency.tolist(),
                'median_mean_across_runs_ms': float(median.mean()),
                'p95_mean_across_runs_ms': float(p95.mean()),
                'peak_allocated_max_mib': float(peak_alloc.max()),
                'peak_allocated_run_values_mib': peak_alloc.tolist(),
                'peak_reserved_max_mib': float(np.nanmax(peak_reserved)),
                'environment': {
                    k: rows[0].get(k)
                    for k in [
                        'profile_protocol_version',
                        'gpu_name',
                        'gpu_total_memory_gib',
                        'compute_capability',
                        'cuda_device_order',
                        'cuda_visible_devices',
                        'python_version',
                        'pytorch_version',
                        'cuda_runtime_version',
                        'cudnn_version',
                        'nvidia_driver_version',
                        'git_commit',
                    ]
                },
            }
            result[benchmark][variant] = item

            print('=' * 80)
            print(benchmark, variant)
            print('parameters:', item['parameters'])
            print(
                'latency mean ± SD across process repeats:',
                f"{item['latency_mean_of_run_means_ms']:.3f} ± {item['latency_sd_of_run_means_ms']:.3f} ms",
            )
            print('run means:', [round(x, 3) for x in latency.tolist()])
            print('peak allocated max:', f"{item['peak_allocated_max_mib']:.2f} MiB")


    result['relative_overhead'] = {}
    for benchmark in ['3DMatch', '3DLoMatch']:
        b = result[benchmark]['baseline']
        m = result[benchmark]['mspki']
        result['relative_overhead'][benchmark] = {
            'parameter_increase_pct': 100.0 * (m['parameters'] / b['parameters'] - 1.0),
            'latency_increase_pct': 100.0 * (
                m['latency_mean_of_run_means_ms'] / b['latency_mean_of_run_means_ms'] - 1.0
            ),
            'peak_allocated_increase_pct': 100.0 * (
                m['peak_allocated_max_mib'] / b['peak_allocated_max_mib'] - 1.0
            ),
        }

    save_path = Path(args.save_json) if args.save_json else root / 'efficiency_summary_final.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSaved:', save_path)


if __name__ == '__main__':
    main()
