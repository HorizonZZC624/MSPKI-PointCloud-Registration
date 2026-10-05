from pathlib import Path
import argparse
import csv
import json
import math
import statistics as st

parser = argparse.ArgumentParser(description="Verify and summarize archived KITTI iteration and efficiency records.")
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3]/"results/additional_evaluation")
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3]/"output/KITTI_additional/summaries")
args = parser.parse_args()
ROOT = args.root
OUTPUT = args.output
OUTPUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)

def mean(values):
    return st.mean(values)

def key(row):
    return tuple(int(row[k]) for k in ('seq', 'src_frame', 'ref_frame'))

report = {'ransac': [], 'efficiency': {}, 'checks': []}
inputs = None
fixed = {}
iteration_rows = []
for iterations in (5000, 10000, 50000):
    folder = ROOT / 'ransac_iterations' / f'iter_{iterations}'
    protocol = json.loads((folder / 'protocol.json').read_text())
    assert protocol['iterations'] == iterations and protocol['completed']
    assert protocol['training_seeds'] == [2026, 3407, 7351]
    assert protocol['estimator_seeds'] == [0, 1, 2, 3, 4]
    assert protocol['budgets'] == [50] and protocol['pairs_per_run'] == 555
    invariant = {k: v for k, v in protocol.items() if k != 'iterations'}
    if inputs is None:
        inputs = invariant
    assert inputs == invariant, 'Protocols differ beyond iteration limit'
    paired = read_csv(folder / 'paired_runs.csv')
    seeds = read_csv(folder / 'training_seed_means.csv')
    summary = read_csv(folder / 'summary.csv')[0]
    assert len(paired) == 15 and len(seeds) == 3
    assert len(list((folder / 'pairs').glob('*.csv'))) == 30
    metrics = [x for x in paired[0] if x not in ('training_seed', 'budget', 'estimator_seed')]
    for run in paired:
        train, estimator = int(run['training_seed']), int(run['estimator_seed'])
        rows = {}
        for method in ('baseline', 'mspki'):
            records = read_csv(folder / 'pairs' / f'{method}_train{train}_top50_ransac{estimator}.csv')
            rows[method] = {key(x): x for x in records}
            assert len(rows[method]) == len(records) == 555
            for r in records:
                assert int(r['correspondences']) == 50
                close(r['IR'], int(r['inliers']) / 50)
                accepted = int(float(r['RRE']) < 5 and float(r['RTE_m']) < 2)
                assert int(r['TR']) == accepted
            immutable = {k: (r['IR'], r['inliers'], r['correspondences']) for k, r in rows[method].items()}
            cache_key = (method, train)
            if cache_key not in fixed:
                fixed[cache_key] = immutable
            assert fixed[cache_key] == immutable, 'Correspondence quality changes across estimator conditions'
        b, m = rows['baseline'], rows['mspki']
        assert set(b) == set(m)
        common = [k for k in b if int(b[k]['TR']) == int(m[k]['TR']) == 1]
        derived = {'pairs': len(b), 'common_success_pairs': len(common)}
        for method, data in rows.items():
            derived[method + '_IR_pct'] = 100 * mean([float(r['IR']) for r in data.values()])
            derived[method + '_TR_pct'] = 100 * mean([int(r['TR']) for r in data.values()])
            derived[method + '_common_RRE_deg'] = mean([float(data[k]['RRE']) for k in common])
            derived[method + '_common_RTE_cm'] = 100 * mean([float(data[k]['RTE_m']) for k in common])
        derived['recovered'] = sum(int(b[k]['TR']) == 0 and int(m[k]['TR']) == 1 for k in b)
        derived['reversed'] = sum(int(b[k]['TR']) == 1 and int(m[k]['TR']) == 0 for k in b)
        for name, value in derived.items():
            close(run[name], value)
    for row in seeds:
        matching = [x for x in paired if x['training_seed'] == row['training_seed']]
        assert len(matching) == 5
        for name in metrics:
            close(row[name], mean([float(x[name]) for x in matching]))
        for metric in ('IR', 'TR'):
            close(row[f'delta_{metric}_pp'], float(row[f'mspki_{metric}_pct']) - float(row[f'baseline_{metric}_pct']))
    for name in metrics + ['delta_IR_pp', 'delta_TR_pp']:
        values = [float(x[name]) for x in seeds]
        close(summary[name + '_mean'], mean(values))
        close(summary[name + '_sample_sd'], st.stdev(values))
    deltas = [float(x['mspki_TR_pct']) - float(x['baseline_TR_pct']) for x in paired]
    entry = {'iterations': iterations, 'summary': {k: float(v) for k, v in summary.items()},
             'seed_means': [{k: float(v) for k, v in x.items()} for x in seeds],
             'positive_TR_runs': sum(x > 0 for x in deltas),
             'negative_TR_runs': sum(x < 0 for x in deltas),
             'positive_common_RRE_runs': sum(float(x['mspki_common_RRE_deg']) < float(x['baseline_common_RRE_deg']) for x in paired),
             'positive_common_RTE_runs': sum(float(x['mspki_common_RTE_cm']) < float(x['baseline_common_RTE_cm']) for x in paired)}
    report['ransac'].append(entry)
    iteration_rows.append({'iterations': iterations, **{k: float(v) for k, v in summary.items()}})

old_path = ROOT / 'budget' / 'summary.csv'
if old_path.exists():
    old = next(x for x in read_csv(old_path) if x['budget'] == '50')
    for k, v in old.items():
        if k in iteration_rows[0]:
            close(v, iteration_rows[0][k])
    report['checks'].append('5k Top-50 matches earlier KITTI budget summary')

efficiency_rows = []
environments = set()
for method in ('baseline', 'mspki'):
    paths = sorted((ROOT / 'kitti_efficiency' / 'runs').glob(f'{method}_run*.json'))
    assert len(paths) == 5
    runs = [json.loads(p.read_text()) for p in paths]
    means = []
    for r in runs:
        assert r['variant'] == method and r['seed'] == 7351
        assert r['re_feature_source'] == 'official' and r['density_keep_ratio'] == 1
        assert r['warmup_pairs'] == 20 and r['profile_pairs'] == len(r['pair_times_ms']) == 100
        close(mean(r['pair_times_ms']), r['latency_mean_ms'])
        close(st.stdev(r['pair_times_ms']), r['latency_sample_sd_ms_within_run'])
        close(st.median(r['pair_times_ms']), r['latency_median_ms'])
        environments.add(tuple(r[k] for k in ('gpu', 'pytorch', 'cuda_runtime', 'cudnn', 'timing_scope')))
        means.append(r['latency_mean_ms'])
    assert len({r['parameters'] for r in runs}) == 1
    entry = {'method': method, 'parameters': runs[0]['parameters'], 'process_repeats': 5,
             'warmup_pairs': 20, 'timed_pairs_per_run': 100,
             'latency_mean_ms': mean(means), 'latency_sample_sd_ms': st.stdev(means),
             'peak_allocated_max_mib': max(r['peak_gpu_memory_mib'] for r in runs),
             'run_means_ms': means, 'peak_values_mib': [r['peak_gpu_memory_mib'] for r in runs],
             'environment': {k: runs[0][k] for k in ('gpu', 'pytorch', 'cuda_runtime', 'cudnn', 'timing_scope')}}
    report['efficiency'][method] = entry
    efficiency_rows.append({k: v for k, v in entry.items() if not isinstance(v, (dict, list))})
assert len(environments) == 1
b, m = report['efficiency']['baseline'], report['efficiency']['mspki']
report['efficiency']['differences'] = {
    'additional_parameters': m['parameters'] - b['parameters'],
    'parameter_increase_pct': 100 * (m['parameters'] / b['parameters'] - 1),
    'additional_latency_ms': m['latency_mean_ms'] - b['latency_mean_ms'],
    'latency_increase_pct': 100 * (m['latency_mean_ms'] / b['latency_mean_ms'] - 1),
    'peak_difference_mib': m['peak_allocated_max_mib'] - b['peak_allocated_max_mib'],
    'peak_increase_pct': 100 * (m['peak_allocated_max_mib'] / b['peak_allocated_max_mib'] - 1)
}
report['checks'] += ['All 90 complete RANSAC pair CSVs verified against summaries (49,950 pose evaluations)',
                     'IR and correspondence counts identical across iteration and estimator conditions',
                     'All ten efficiency JSONs verified against 100 per-pair timing values each',
                     'No smoke-test records included in efficiency aggregation']

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUTPUT / 'kitti_efficiency_summary.csv', efficiency_rows)
write_csv(OUTPUT / 'iteration_control_summary.csv', iteration_rows)
(OUTPUT / 'iterations_efficiency_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
