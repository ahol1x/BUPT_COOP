#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-cifar100}"
SCENARIO="${SCENARIO:-B0-Inc10}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FAST_DEV_RUN="${FAST_DEV_RUN:-0}"
SEEDS="${SEEDS:-1991 1993 1995}"
STRATEGIES="${STRATEGIES:-adaptive prompt_only tae_only adapter_each_task mote_fusion all_combined finetune}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/daac_compare}"
DATA_DIR="${DATA_DIR:-data}"
DEVICE="${DEVICE:-auto}"
STRICT="${STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"
INCREMENT="${INCREMENT:-10}"
INIT_CLASSES="${INIT_CLASSES:-0}"
MAX_TASKS="${MAX_TASKS:-}"
PRESTUDY_BATCHES="${PRESTUDY_BATCHES:-2}"
LR="${LR:-0.001}"
EMBED_DIM="${EMBED_DIM:-64}"
DEPTH="${DEPTH:-3}"
NUM_HEADS="${NUM_HEADS:-4}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast_dev_run)
      FAST_DEV_RUN=1
      shift
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

COMPARE_ROOT="${OUTPUT_ROOT}/${DATASET}/${SCENARIO}"
RAW_ROOT="${COMPARE_ROOT}/.raw_runs"
mkdir -p "$COMPARE_ROOT" "$RAW_ROOT"

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "WARNING: CUDA is unavailable. CPU smoke/basic runs are allowed, but timing and memory are not GPU-comparable." >&2
fi

cat > "${COMPARE_ROOT}/command_used.txt" <<EOF
DATASET=${DATASET}
SCENARIO=${SCENARIO}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
FAST_DEV_RUN=${FAST_DEV_RUN}
SEEDS=${SEEDS}
STRATEGIES=${STRATEGIES}
OUTPUT_ROOT=${OUTPUT_ROOT}
DATA_DIR=${DATA_DIR}
DEVICE=${DEVICE}
STRICT=${STRICT}
DRY_RUN=${DRY_RUN}
INCREMENT=${INCREMENT}
INIT_CLASSES=${INIT_CLASSES}
MAX_TASKS=${MAX_TASKS}
PRESTUDY_BATCHES=${PRESTUDY_BATCHES}
LR=${LR}
EMBED_DIM=${EMBED_DIM}
DEPTH=${DEPTH}
NUM_HEADS=${NUM_HEADS}
ADAPTER_BOTTLENECK=${ADAPTER_BOTTLENECK}
EOF

run_one() {
  local strategy="$1"
  local seed="$2"
  local dest="${COMPARE_ROOT}/${strategy}/seed_${seed}"
  local raw="${RAW_ROOT}/${strategy}/seed_${seed}"
  rm -rf "$dest" "$raw"
  mkdir -p "$dest" "$raw"

  local cmd=(
    python run_daac.py
    --strategy "$strategy"
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
    --output-dir "$raw"
    --seeds "$seed"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --increment "$INCREMENT"
    --init-classes "$INIT_CLASSES"
    --prestudy-batches "$PRESTUDY_BATCHES"
    --lr "$LR"
    --device "$DEVICE"
    --embed-dim "$EMBED_DIM"
    --depth "$DEPTH"
    --num-heads "$NUM_HEADS"
    --adapter-bottleneck "$ADAPTER_BOTTLENECK"
  )
  if [[ -n "$MAX_TASKS" ]]; then
    cmd+=(--max-tasks "$MAX_TASKS")
  fi
  if [[ "$FAST_DEV_RUN" == "1" ]]; then
    cmd+=(--fast_dev_run)
  fi

  printf '%q ' "${cmd[@]}" > "${dest}/command.txt"
  printf '\n' >> "${dest}/command.txt"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN ${strategy} seed_${seed}: ${cmd[*]}"
    python - "$dest/args_used.json" "$strategy" "$seed" "$DATASET" "$SCENARIO" "$FAST_DEV_RUN" <<'PY'
import json, sys
path, strategy, seed, dataset, scenario, fast = sys.argv[1:]
payload = {
    "strategy": strategy,
    "seed": int(seed),
    "dataset": dataset,
    "scenario": scenario,
    "fast_dev_run": fast == "1",
    "dry_run": True,
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
PY
    return 0
  fi

  echo "RUNNING strategy=${strategy} seed=${seed}"
  "${cmd[@]}" > "${dest}/stdout.log" 2> "${dest}/stderr.log"
  local status=$?
  cat "${dest}/stdout.log" "${dest}/stderr.log" > "${dest}/train.log"

  local raw_run=""
  raw_run="$(find "$raw/daac" -path "*/${strategy}/${seed}" -type d 2>/dev/null | head -n 1 || true)"
  if [[ -n "$raw_run" ]]; then
    [[ -f "${raw_run}/metrics.csv" ]] && cp "${raw_run}/metrics.csv" "${dest}/metrics.csv"
    [[ -f "${raw_run}/summary.json" ]] && cp "${raw_run}/summary.json" "${dest}/summary.json"
    [[ -f "${raw_run}/run_config.json" ]] && cp "${raw_run}/run_config.json" "${dest}/args_used.json"
  fi

  python - "$dest/run_status.json" "$strategy" "$seed" "$status" <<'PY'
import json, sys
path, strategy, seed, status = sys.argv[1:]
payload = {
    "strategy": strategy,
    "seed": int(seed),
    "exit_code": int(status),
    "passed": int(status) == 0,
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
PY

  if [[ "$status" -ne 0 ]]; then
    echo "WARNING: ${strategy} seed_${seed} failed with exit code ${status}. Logs: ${dest}/train.log" >&2
    return "$status"
  fi
  if [[ ! -f "${dest}/metrics.csv" || ! -f "${dest}/summary.json" ]]; then
    echo "WARNING: ${strategy} seed_${seed} finished but metrics/summary were not found in ${dest}" >&2
    return 3
  fi
  echo "PASSED ${strategy} seed_${seed}"
  return 0
}

overall_status=0
for strategy in $STRATEGIES; do
  for seed in $SEEDS; do
    run_one "$strategy" "$seed"
    status=$?
    if [[ "$status" -ne 0 ]]; then
      overall_status="$status"
      if [[ "$STRICT" == "1" ]]; then
        exit "$status"
      fi
    fi
  done
done

exit "$overall_status"
