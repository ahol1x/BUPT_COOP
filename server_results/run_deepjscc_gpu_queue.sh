#!/usr/bin/env bash
set -u

PROJECT_DIR="$HOME/projects/BUPT/Chris/semantic_comm/Deep-JSCC-PyTorch-main"
ENV_PATH="$HOME/envs/bupt-cil/bin/activate"
QUEUE="$PROJECT_DIR/jobs_deepjscc_queue.txt"
LOCK="$PROJECT_DIR/jobs_deepjscc_queue.lock"
THRESHOLD_MB=1000

cd "$PROJECT_DIR"
source "$ENV_PATH"

wait_gpu_free() {
    local gpu=$1
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')
        echo "$(date) | GPU $gpu memory=${mem}MiB"
        if [ "$mem" -lt "$THRESHOLD_MB" ]; then
            return 0
        fi
        sleep 120
    done
}

pop_job() {
    local line
    {
        flock -x 9
        line=$(python - "$QUEUE" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("")
    raise SystemExit

lines = p.read_text().splitlines()
job = ""
remaining = []

used = False
for line in lines:
    stripped = line.strip()
    if not used and stripped and not stripped.startswith("#"):
        job = stripped
        used = True
    else:
        remaining.append(line)

p.write_text("\n".join(remaining) + ("\n" if remaining else ""))
print(job)
PY
)
    } 9>"$LOCK"

    echo "$line"
}

worker() {
    local gpu=$1
    echo "Worker for GPU $gpu started at $(date)"

    while true; do
        wait_gpu_free "$gpu"

        job=$(pop_job)
        if [ -z "$job" ]; then
            echo "GPU $gpu: no jobs left at $(date)"
            break
        fi

        read -r channel snr ratio <<< "$job"

        channel_lower=$(echo "$channel" | tr '[:upper:]' '[:lower:]')
        ratio_tag=$(echo "$ratio" | tr -d '/')

        out_dir="./out_cifar10_${channel_lower}_snr${snr}_r${ratio_tag}"
        log_file="log_gpu${gpu}_${channel_lower}_snr${snr}_r${ratio_tag}.txt"

        echo "GPU $gpu starting job: channel=$channel snr=$snr ratio=$ratio"
        echo "Output: $out_dir"
        echo "Log: $log_file"
        echo "Start time: $(date)"

        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --dataset cifar10 \
            --channel "$channel" \
            --snr_list "$snr" \
            --ratio_list "$ratio" \
            --out "$out_dir" \
            > "$log_file" 2>&1

        echo "GPU $gpu finished job: channel=$channel snr=$snr ratio=$ratio"
        echo "Finish time: $(date)"
    done
}

worker 1 > worker_gpu1.log 2>&1 &
PID1=$!

worker 2 > worker_gpu2.log 2>&1 &
PID2=$!

worker 3 > worker_gpu3.log 2>&1 &
PID3=$!

echo "Started GPU workers:"
echo "GPU1 worker PID=$PID1"
echo "GPU2 worker PID=$PID2"
echo "GPU3 worker PID=$PID3"

wait $PID1 $PID2 $PID3

echo "All queued jobs finished at $(date)"
