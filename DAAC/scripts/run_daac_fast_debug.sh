#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run_daac.py \
  --fast_dev_run \
  --dataset synthetic \
  --strategies adaptive prompt_only tae_only adapter_each_task mote_fusion all_combined finetune \
  --seeds 1991 \
  --epochs 1 \
  --batch-size 16 \
  --increment 2 \
  --max-tasks 3 \
  --output-dir outputs
