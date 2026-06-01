#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

STRICT="${STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/daac_compare}"
STRATEGIES="${STRATEGIES:-adaptive prompt_only tae_only adapter_each_task mote_fusion all_combined finetune}"
SEED="${SEED:-1993}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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

failed=0
for strategy in $STRATEGIES; do
  DATASET=synthetic \
  SCENARIO=smoke \
  EPOCHS=1 \
  BATCH_SIZE=8 \
  FAST_DEV_RUN=1 \
  SEEDS="$SEED" \
  STRATEGIES="$strategy" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  INCREMENT=2 \
  MAX_TASKS=2 \
  PRESTUDY_BATCHES=1 \
  EMBED_DIM=24 \
  DEPTH=2 \
  NUM_HEADS=4 \
  ADAPTER_BOTTLENECK=6 \
  STRICT="$STRICT" \
  DRY_RUN="$DRY_RUN" \
    bash scripts/run_daac_basic_compare.sh --fast_dev_run
  status=$?

  metrics_path="${OUTPUT_ROOT}/synthetic/smoke/${strategy}/seed_${SEED}/metrics.csv"
  if [[ "$status" -eq 0 && ( "$DRY_RUN" == "1" || -f "$metrics_path" ) ]]; then
    echo "PASSED ${strategy}"
  else
    echo "FAILED ${strategy}"
    failed=1
    if [[ "$STRICT" == "1" ]]; then
      exit "$status"
    fi
  fi
done

exit "$failed"
