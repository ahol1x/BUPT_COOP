#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-cifar100}"
SCENARIO="${SCENARIO:-B0-Inc10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/daac_compare}"
STRICT="${STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-0}"

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
    --fast_dev_run)
      FAST_DEV_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

FIG_DIR="${OUTPUT_ROOT}/${DATASET}/${SCENARIO}/figures"
REPORT_PATH="${OUTPUT_ROOT}/${DATASET}/${SCENARIO}/DAAC_basic_comparison_report.md"

basic_cmd=(bash scripts/run_daac_basic_compare.sh)
if [[ "$FAST_DEV_RUN" == "1" ]]; then
  basic_cmd+=(--fast_dev_run)
fi
if [[ "$STRICT" == "1" ]]; then
  basic_cmd+=(--strict)
fi

aggregate_cmd=(python scripts/aggregate_daac_results.py --root "$OUTPUT_ROOT" --dataset "$DATASET" --scenario "$SCENARIO")
plot_cmd=(python scripts/plot_daac_comparison.py --root "$OUTPUT_ROOT" --dataset "$DATASET" --scenario "$SCENARIO" --out_dir "$FIG_DIR")
report_cmd=(python scripts/generate_daac_report.py --root "$OUTPUT_ROOT" --dataset "$DATASET" --scenario "$SCENARIO" --figures-dir "$FIG_DIR")

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN:"
  printf '  '; printf '%q ' "${basic_cmd[@]}"; printf '\n'
  printf '  '; printf '%q ' "${aggregate_cmd[@]}"; printf '\n'
  printf '  '; printf '%q ' "${plot_cmd[@]}"; printf '\n'
  printf '  '; printf '%q ' "${report_cmd[@]}"; printf '\n'
  exit 0
fi

STRICT="$STRICT" FAST_DEV_RUN="$FAST_DEV_RUN" DATASET="$DATASET" SCENARIO="$SCENARIO" OUTPUT_ROOT="$OUTPUT_ROOT" "${basic_cmd[@]}"
basic_status=$?
if [[ "$basic_status" -ne 0 ]]; then
  echo "WARNING: comparison runner exited with ${basic_status}; continuing to aggregate available runs." >&2
  if [[ "$STRICT" == "1" ]]; then
    exit "$basic_status"
  fi
fi

"${aggregate_cmd[@]}" || exit $?
"${plot_cmd[@]}" || exit $?
"${report_cmd[@]}" || exit $?

echo "Comparison completed."
echo "Aggregated CSV:"
echo "  ${OUTPUT_ROOT}/${DATASET}/${SCENARIO}/aggregated_results.csv"
echo "  ${OUTPUT_ROOT}/${DATASET}/${SCENARIO}/aggregated_summary.csv"
echo "Figures:"
echo "  ${FIG_DIR}"
echo "Report:"
echo "  ${REPORT_PATH}"
