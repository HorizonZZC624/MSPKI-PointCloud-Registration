from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Audit paper test manifests and saved-pair counts.')
    parser.add_argument('--search_root', required=True)
    args = parser.parse_args()

    failures = 0
    manifests = list(Path(args.search_root).rglob('manifest.json'))
    for path in manifests:
        with open(path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        expected = int(m.get('expected_pairs', -1))
        saved = int(m.get('saved_pairs', -1))
        actual = len(glob.glob(str(path.parent / '*' / '*.npz')))
        ok = bool(m.get('completed', False)) and expected == saved == actual and expected >= 0
        status = 'OK' if ok else 'FAIL'
        print(
            f"[{status}] {path.parent} | variant={m.get('variant')} seed={m.get('seed')} "
            f"benchmark={m.get('benchmark')} expected={expected} saved={saved} actual={actual}"
        )
        failures += int(not ok)

    if not manifests:
        raise SystemExit('No manifest.json files found.')
    if failures:
        raise SystemExit(f'{failures} run(s) failed integrity checks.')
    print(f'All {len(manifests)} runs passed integrity checks.')


if __name__ == '__main__':
    main()
