#!/usr/bin/env bash
set -euo pipefail














PYTHON_EXE="${PYTHON_EXE:-python}"
MASK_SEED=9017
NUM_WORKERS=0
DRY_RUN=0
KEEP_RATIOS=(1.00 0.75 0.50 0.25)

usage() {
    cat <<'EOF'
Usage:
  ./run_density_stress.sh [options]

Options:
  --dry-run                  Print planned runs without executing them
  --python PATH              Python executable (default: current environment's python)
  --mask-seed N              Density mask seed (default: 9017)
  --num-workers N            DataLoader workers (default: 0)
  --keep-ratios R1 R2 ...    Keep ratios, e.g. 1.00 0.75 0.50 0.25
  -h, --help                 Show this help

Examples:
  ./run_density_stress.sh --dry-run
  ./run_density_stress.sh
  ./run_density_stress.sh --num-workers 8
  ./run_density_stress.sh --keep-ratios 1.00 0.75 0.50 0.25
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "ERROR: --python requires a value" >&2; exit 2; }
            PYTHON_EXE="$2"
            shift 2
            ;;
        --mask-seed)
            [[ $# -ge 2 ]] || { echo "ERROR: --mask-seed requires a value" >&2; exit 2; }
            MASK_SEED="$2"
            shift 2
            ;;
        --num-workers)
            [[ $# -ge 2 ]] || { echo "ERROR: --num-workers requires a value" >&2; exit 2; }
            NUM_WORKERS="$2"
            shift 2
            ;;
        --keep-ratios)
            shift
            KEEP_RATIOS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                KEEP_RATIOS+=("$1")
                shift
            done
            [[ ${#KEEP_RATIOS[@]} -gt 0 ]] || {
                echo "ERROR: --keep-ratios requires at least one ratio" >&2
                exit 2
            }
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="$PROJECT_ROOT/datasets/KITTI"
METADATA_ROOT="$PROJECT_ROOT/data/KITTI/metadata"
OUTPUT_ROOT="$PROJECT_ROOT/output/KITTI_mspki"

TEST_SCRIPT="$SCRIPT_DIR/test.py"
EVAL_SCRIPT="$SCRIPT_DIR/eval.py"
AGGREGATE_SCRIPT="$SCRIPT_DIR/aggregate_density_stress.py"




if ! command -v "$PYTHON_EXE" >/dev/null 2>&1 && [[ ! -x "$PYTHON_EXE" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_EXE" >&2
    exit 1
fi

for path in \
    "$DATASET_ROOT" \
    "$METADATA_ROOT" \
    "$TEST_SCRIPT" \
    "$EVAL_SCRIPT" \
    "$AGGREGATE_SCRIPT"
do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Missing input: $path" >&2
        exit 1
    fi
done


for ratio in "${KEEP_RATIOS[@]}"; do
    "$PYTHON_EXE" - "$ratio" <<'PY'
import sys
r = float(sys.argv[1])
if not (0.0 < r <= 1.0):
    raise SystemExit(f"Keep ratios must be in (0, 1], got {r}")
PY
done

echo "======================================================"
echo "KITTI density stress test"
echo "======================================================"
echo "Python        : $PYTHON_EXE"
echo "Project root  : $PROJECT_ROOT"
echo "Dataset root  : $DATASET_ROOT"
echo "Metadata root : $METADATA_ROOT"
echo "Output root   : $OUTPUT_ROOT"
echo "Mask seed     : $MASK_SEED"
echo "Num workers   : $NUM_WORKERS"
echo "Keep ratios   : ${KEEP_RATIOS[*]}"
echo "Dry run       : $DRY_RUN"
echo "======================================================"

VARIANTS=(baseline mspki)
TRAINING_SEEDS=(2026 3407 7351)

for ratio in "${KEEP_RATIOS[@]}"; do
    ratio_text="$("$PYTHON_EXE" - "$ratio" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}")
PY
)"
    ratio_tag="$("$PYTHON_EXE" - "$ratio" <<'PY'
import sys
print(int(round(100 * float(sys.argv[1]))))
PY
)"
    run_tag="$(printf 'density_src_r%03d_m%s' "$ratio_tag" "$MASK_SEED")"

    for variant in "${VARIANTS[@]}"; do
        for training_seed in "${TRAINING_SEEDS[@]}"; do

            checkpoint="$OUTPUT_ROOT/$variant/re_official/seed_${training_seed}/snapshots/best.pth.tar"

            if [[ ! -f "$checkpoint" ]]; then
                echo "ERROR: Missing checkpoint: $checkpoint" >&2
                exit 1
            fi

            run_dir="$OUTPUT_ROOT/$variant/re_official/seed_${training_seed}/$run_tag"
            feature_dir="$run_dir/features"
            manifest_file="$feature_dir/manifest.json"
            summary_file="$run_dir/registration/summary_fhp.json"

            if (( DRY_RUN == 1 )); then
                echo "DRY RUN: $variant seed=$training_seed keep=$ratio_text tag=$run_tag"
                continue
            fi

            features_complete=0

            if [[ -f "$manifest_file" ]]; then
                manifest_status="$("$PYTHON_EXE" - "$manifest_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)

ok = (
    bool(m.get("completed", False))
    and int(m.get("saved_pairs", -1)) == 555
    and int(m.get("expected_pairs", -1)) == 555
)
print("1" if ok else "0")
PY
)"
                if [[ "$manifest_status" == "1" ]]; then
                    features_complete=1
                else
                    echo "ERROR: Incomplete features require inspection: $feature_dir" >&2
                    exit 1
                fi

            elif [[ -d "$feature_dir" ]] && find "$feature_dir" -maxdepth 1 -type f -name '*.npz' -print -quit | grep -q .; then
                echo "ERROR: Feature files exist without a completed manifest: $feature_dir" >&2
                exit 1
            fi

            if [[ -f "$summary_file" ]]; then
                summary_pairs="$("$PYTHON_EXE" - "$summary_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    s = json.load(f)

print(int(s.get("num_pairs", -1)))
PY
)"
                if [[ "$features_complete" -ne 1 || "$summary_pairs" -ne 555 ]]; then
                    echo "ERROR: Existing summary does not match a completed 555-pair run: $summary_file" >&2
                    exit 1
                fi

                echo "Already complete: $variant seed=$training_seed keep=$ratio_text"
                continue
            fi

            echo
            echo "======================================================"
            echo "Variant       : $variant"
            echo "Training seed : $training_seed"
            echo "Keep ratio    : $ratio_text"
            echo "Run tag       : $run_tag"
            echo "======================================================"

            if [[ "$features_complete" -ne 1 ]]; then
                echo "[1/2] Extracting density-stress features..."

                "$PYTHON_EXE" "$TEST_SCRIPT" \
                    --variant "$variant" \
                    --seed "$training_seed" \
                    --dataset_root "$DATASET_ROOT" \
                    --metadata_root "$METADATA_ROOT" \
                    --re_feature_source official \
                    --run_tag "$run_tag" \
                    --num_workers "$NUM_WORKERS" \
                    --snapshot "$checkpoint" \
                    --density_keep_ratio "$ratio_text" \
                    --density_mask_seed "$MASK_SEED"
            else
                echo "[1/2] Features already complete; skipping extraction."
            fi

            echo "[2/2] Evaluating registration..."

            "$PYTHON_EXE" "$EVAL_SCRIPT" \
                --variant "$variant" \
                --seed "$training_seed" \
                --dataset_root "$DATASET_ROOT" \
                --metadata_root "$METADATA_ROOT" \
                --re_feature_source official \
                --run_tag "$run_tag"

        done
    done
done

if (( DRY_RUN == 0 )); then
    echo
    echo "======================================================"
    echo "Aggregating KITTI density-stress results"
    echo "======================================================"

    ratio_args=()
    for ratio in "${KEEP_RATIOS[@]}"; do
        ratio_args+=("$("$PYTHON_EXE" - "$ratio" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}")
PY
)")
    done

    "$PYTHON_EXE" "$AGGREGATE_SCRIPT" \
        --output_root "$OUTPUT_ROOT" \
        --mask_seed "$MASK_SEED" \
        --keep_ratios "${ratio_args[@]}"

    echo
    echo "======================================================"
    echo "KITTI density stress test COMPLETE"
    echo "======================================================"
fi
