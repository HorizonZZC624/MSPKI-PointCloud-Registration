#!/usr/bin/env bash








set -euo pipefail




PYTHON_EXE="${PYTHON_EXE:-python}"
TRAINING_SEEDS=(2026 3407 7351)
BUDGETS=(50 0)
ESTIMATOR_SEEDS=(0 1 2 3 4)
BENCHMARK="${BENCHMARK:-3DLoMatch}"
DRY_RUN=0



if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

if [[ "$BENCHMARK" != "3DMatch" && "$BENCHMARK" != "3DLoMatch" ]]; then
    echo "ERROR: BENCHMARK must be 3DMatch or 3DLoMatch, got: $BENCHMARK" >&2
    exit 1
fi



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/reproduction_ransac}"
EVAL_SCRIPT="$PROJECT_ROOT/experiments/3DMatch/eval.py"
AGGREGATE_SCRIPT="$PROJECT_ROOT/experiments/3DMatch/paper_eval_suite/aggregate_ransac_estimator_seeds.py"




if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_EXE" >&2
    echo "Current PATH: $PATH" >&2
    exit 1
fi

for path in "$OUTPUT_ROOT" "$EVAL_SCRIPT" "$AGGREGATE_SCRIPT"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Missing input: $path" >&2
        exit 1
    fi
done

OPEN3D_VERSION="$($PYTHON_EXE -c 'import open3d; print(open3d.__version__)')" || {
    echo "ERROR: Could not import Open3D from the selected Python environment." >&2
    exit 1
}

echo "Open3D version: $OPEN3D_VERSION"
echo "Python: $($PYTHON_EXE -c 'import sys; print(sys.executable)')"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Benchmark: $BENCHMARK"

contains_seed_7351=0
for s in "${TRAINING_SEEDS[@]}"; do
    if [[ "$s" == "7351" ]]; then
        contains_seed_7351=1
        break
    fi
done

if [[ "$OPEN3D_VERSION" != "0.18.0" && "$contains_seed_7351" -eq 0 ]]; then
    echo "WARNING: The manuscript used Open3D 0.18.0. Include seed 7351 in this environment before combining the new runs with the old RANSAC table." >&2
fi


validate_summary() {
    local summary_file="$1"
    local expected_method_tag="$2"
    local expected_estimator_seed="$3"
    local expected_training_seed="$4"
    local expected_variant="$5"

    "$PYTHON_EXE" - "$summary_file" "$expected_method_tag" "$expected_estimator_seed" "$expected_training_seed" "$expected_variant" <<'PY'
import json
import sys

path, method_tag, estimator_seed, training_seed, variant = sys.argv[1:]
with open(path, 'r', encoding='utf-8') as f:
    summary = json.load(f)

checks = {
    'method_tag': (summary.get('method_tag'), method_tag),
    'estimator_seed': (summary.get('estimator_seed'), int(estimator_seed)),
    'seed': (summary.get('seed'), int(training_seed)),
    'variant': (summary.get('variant'), variant),
}

bad = [(k, got, exp) for k, (got, exp) in checks.items() if got != exp]
if bad:
    print(f"ERROR: Existing RANSAC summary does not match its run: {path}", file=sys.stderr)
    for k, got, exp in bad:
        print(f"  {k}: got={got!r}, expected={exp!r}", file=sys.stderr)
    sys.exit(1)
PY
}




for training_seed in "${TRAINING_SEEDS[@]}"; do
    for variant in baseline mspki; do
        feature_dir="$OUTPUT_ROOT/$variant/re_official/seed_${training_seed}/features/$BENCHMARK"
        manifest="$feature_dir/manifest.json"

        if [[ ! -f "$manifest" ]]; then
            echo "ERROR: Missing feature manifest: $manifest" >&2
            exit 1
        fi

        for budget in "${BUDGETS[@]}"; do
            if (( budget < 0 )); then
                echo "ERROR: Use budget 0 to represent all correspondences." >&2
                exit 1
            fi

            for estimator_seed in "${ESTIMATOR_SEEDS[@]}"; do
                if (( budget > 0 )); then
                    method_tag="ransac_n${budget}_s${estimator_seed}"
                else
                    method_tag="ransac_s${estimator_seed}"
                fi

                summary_file="$OUTPUT_ROOT/$variant/re_official/seed_${training_seed}/registration/$BENCHMARK/summary_${method_tag}.json"

                args=(
                    "$EVAL_SCRIPT"
                    --variant "$variant"
                    --seed "$training_seed"
                    --re_feature_source official
                    --output_root "$OUTPUT_ROOT"
                    --benchmark "$BENCHMARK"
                    --method ransac
                    --estimator_seed "$estimator_seed"
                )

                if (( budget > 0 )); then
                    args+=(--num_corr "$budget")
                fi

                if (( DRY_RUN == 1 )); then
                    echo "DRY RUN: $variant seed=$training_seed budget=$budget estimator_seed=$estimator_seed"
                    printf '  %q' "$PYTHON_EXE" "${args[@]}"
                    printf '\n'
                elif [[ -f "$summary_file" ]]; then
                    validate_summary \
                        "$summary_file" \
                        "$method_tag" \
                        "$estimator_seed" \
                        "$training_seed" \
                        "$variant"
                    echo "Already complete: $variant seed=$training_seed budget=$budget estimator_seed=$estimator_seed"
                else
                    echo "============================================================"
                    echo "Running: variant=$variant training_seed=$training_seed budget=$budget estimator_seed=$estimator_seed"
                    echo "============================================================"
                    "$PYTHON_EXE" "${args[@]}" || {
                        echo "ERROR: RANSAC failed: $variant/$training_seed/$BENCHMARK/budget=$budget/estimator_seed=$estimator_seed" >&2
                        exit 1
                    }
                fi
            done
        done
    done

    if (( DRY_RUN == 0 )); then
        baseline_registration="$OUTPUT_ROOT/baseline/re_official/seed_${training_seed}/registration/$BENCHMARK"
        mspki_registration="$OUTPUT_ROOT/mspki/re_official/seed_${training_seed}/registration/$BENCHMARK"
        summary_dir="$OUTPUT_ROOT/ransac_training_seed_analysis/$BENCHMARK/seed_${training_seed}"

        echo "============================================================"
        echo "Aggregating training seed $training_seed"
        echo "============================================================"

        "$PYTHON_EXE" "$AGGREGATE_SCRIPT" \
            --baseline_registration_dir "$baseline_registration" \
            --mspki_registration_dir "$mspki_registration" \
            --output_dir "$summary_dir" || {
                echo "ERROR: Aggregation failed for training seed $training_seed" >&2
                exit 1
            }
    fi
done

echo "============================================================"
echo "All RANSAC training-seed evaluations completed successfully."
echo "============================================================"
