#!/usr/bin/env bash
set -euo pipefail

cd /workspace/BUPT/MoTE
mkdir -p /workspace/runs/mote_cub_5seed_logs
mkdir -p /workspace/runs/mote_cub_5seed_status

for SEED in 1991 1992 1993 1994 1995; do
  CONFIG="exps/cub/vast_5seeds/mote_in21k_cub_seed${SEED}.json"
  LOG="/workspace/runs/mote_cub_5seed_logs/mote_cub_seed${SEED}_$(date +%F_%H%M).log"

  echo "============================================================"
  echo "Running MoTE CUB B0-Inc20 seed ${SEED}"
  echo "Config: ${CONFIG}"
  echo "Log: ${LOG}"
  echo "============================================================"

  echo "START seed ${SEED} $(date)" | tee -a /workspace/runs/mote_cub_5seed_status/status.txt

  CUDA_VISIBLE_DEVICES=0 python -u main.py --config "${CONFIG}" 2>&1 | tee "${LOG}"

  echo "DONE seed ${SEED} $(date)" | tee -a /workspace/runs/mote_cub_5seed_status/status.txt
done

echo "ALL DONE $(date)" | tee -a /workspace/runs/mote_cub_5seed_status/status.txt
