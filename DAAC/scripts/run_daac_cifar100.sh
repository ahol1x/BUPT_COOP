#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run_daac.py \
  --dataset cifar100 \
  --strategies adaptive prompt_only tae_only adapter_each_task mote_fusion all_combined \
  --seeds 1991 1993 1995 \
  --init-classes 0 \
  --increment 10 \
  --epochs 20 \
  --batch-size 64 \
  --prestudy-batches 2 \
  --embed-dim 64 \
  --depth 3 \
  --num-heads 4 \
  --adapter-bottleneck 16 \
  --output-dir outputs
