#!/usr/bin/env bash
set -euo pipefail

TAG="$1"
PID_FILE="$2"
CURRENT_LOG="$3"
CONFIG_DIR="$4"
RUN_DIR="$5"

source /workspace/.bupt_env.sh
cd /workspace/BUPT/MoTE
mkdir -p "$RUN_DIR"

STATUS="$RUN_DIR/status.txt"
CURRENT_PID="$(cat "$PID_FILE")"

echo "WAIT ${TAG} seed1993 PID=${CURRENT_PID} $(date)" | tee -a "$STATUS"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 30
done

if ! grep -q "Forgetting (CNN):" "$CURRENT_LOG"; then
    echo "FAILED ${TAG} seed1993 did not finish correctly $(date)" | tee -a "$STATUS"
    exit 2
fi

echo "DONE ${TAG} seed1993 $(date)" | tee -a "$STATUS"

for SEED in 1991 1992 1994 1995; do
    CONFIG="${CONFIG_DIR}/mote_in21k_seed${SEED}.json"
    LOG="${RUN_DIR}/seed${SEED}.log"

    echo "START ${TAG} seed${SEED} $(date)" | tee -a "$STATUS"

    python -u main.py --config "$CONFIG" > "$LOG" 2>&1

    if ! grep -q "Forgetting (CNN):" "$LOG"; then
        echo "FAILED ${TAG} seed${SEED} missing final metrics $(date)" | tee -a "$STATUS"
        exit 3
    fi

    echo "DONE ${TAG} seed${SEED} $(date)" | tee -a "$STATUS"
done

echo "ALL DONE ${TAG} $(date)" | tee -a "$STATUS"
