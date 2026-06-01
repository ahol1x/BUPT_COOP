#!/usr/bin/env bash
set -e
set -o pipefail

REPO_DIR="/home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT"
CONDA_SH="${CONDA_SH:-/home/kennycao12/miniconda3/etc/profile.d/conda.sh}"
EXPECTED_PYTHON_PREFIX="/home/kennycao12/miniconda3/envs/bupt-cil/"
RUN_NAME="step1_mote_cifar100_b0inc10_seed1993"
CONFIG="exps/${RUN_NAME}.json"
OUTPUT_DIR="outputs/${RUN_NAME}"
WRAPPER_LOG="logs/${RUN_NAME}.log"
REPO_LOG="logs/mos/cifar224/0/10/step1_1993_vit_base_patch16_224_mos.log"
GPU_MEMORY_SAMPLES="${OUTPUT_DIR}/gpu_memory_samples.csv"
WALL_CLOCK_SECONDS_FILE="${OUTPUT_DIR}/wall_clock_seconds.txt"
METRICS_JSON="${OUTPUT_DIR}/metrics_summary.json"

cd "${REPO_DIR}"
mkdir -p logs results "${OUTPUT_DIR}" .cache/torch .cache/huggingface
: > "${WRAPPER_LOG}"
exec > >(tee -a "${WRAPPER_LOG}") 2>&1

echo "Running ${RUN_NAME}"
echo "Repository: ${REPO_DIR}"
echo "Config: ${CONFIG}"
echo "Configured repo method: mos"
echo "Audit note: no literal MoTE implementation was found in this mirror; see docs/step1_mote_repo_audit.md."

if [ -f "${CONDA_SH}" ]; then
    # shellcheck source=/home/kennycao12/miniconda3/etc/profile.d/conda.sh
    source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda was not found and ${CONDA_SH} does not exist."
    exit 1
fi

conda activate bupt-cil

echo "which python:"
which python
echo "python --version:"
python --version

PYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
case "${PYTHON_EXECUTABLE}" in
    "${EXPECTED_PYTHON_PREFIX}"*) ;;
    *)
        echo "ERROR: python executable is ${PYTHON_EXECUTABLE}"
        echo "Expected it to be under ${EXPECTED_PYTHON_PREFIX}"
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="${PWD}/.cache/torch"
export HF_HOME="${PWD}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PWD}/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY}"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing config ${CONFIG}"
    exit 1
fi

: > "${GPU_MEMORY_SAMPLES}"
GPU_MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,memory.used --format=csv,noheader,nounits -l 1 > "${GPU_MEMORY_SAMPLES}" &
    GPU_MONITOR_PID="$!"
    echo "Started GPU memory sampler with pid ${GPU_MONITOR_PID}"
else
    echo "nvidia-smi not found; peak GPU memory will be unavailable."
fi

cleanup_gpu_sampler() {
    if [ -n "${GPU_MONITOR_PID}" ]; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
        GPU_MONITOR_PID=""
    fi
}
trap cleanup_gpu_sampler EXIT

SECONDS=0
TRAIN_STATUS=0
python main.py --config="${CONFIG}" || TRAIN_STATUS="$?"
WALL_CLOCK_SECONDS="${SECONDS}"
printf "%s\n" "${WALL_CLOCK_SECONDS}" > "${WALL_CLOCK_SECONDS_FILE}"
cleanup_gpu_sampler

PEAK_GPU_MEMORY_MIB=""
if [ -s "${GPU_MEMORY_SAMPLES}" ]; then
    PEAK_GPU_MEMORY_MIB="$(awk -F',' 'BEGIN { max = "" } { gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3 != "" && (max == "" || $3 + 0 > max + 0)) max = $3 + 0 } END { if (max != "") print max }' "${GPU_MEMORY_SAMPLES}")"
    if [ -n "${PEAK_GPU_MEMORY_MIB}" ]; then
        echo "Peak GPU memory MiB: ${PEAK_GPU_MEMORY_MIB}"
    fi
fi

python scripts/extract_metrics.py \
    --run-name "${RUN_NAME}" \
    --config "${CONFIG}" \
    --wrapper-log "${WRAPPER_LOG}" \
    --repo-log "${REPO_LOG}" \
    --gpu-memory-samples "${GPU_MEMORY_SAMPLES}" \
    --wall-clock-seconds "${WALL_CLOCK_SECONDS_FILE}" \
    --output "${METRICS_JSON}" \
    --dataset-requested "CIFAR100" \
    --dataset-config-key "cifar224" \
    --split "B0-Inc10" \
    --seed 1993

echo "Wall-clock seconds: ${WALL_CLOCK_SECONDS}"
echo "Wrapper log: ${WRAPPER_LOG}"
echo "Repo log: ${REPO_LOG}"
echo "Metrics summary: ${METRICS_JSON}"

if [ "${TRAIN_STATUS}" -ne 0 ]; then
    echo "Training command failed with exit status ${TRAIN_STATUS}"
    exit "${TRAIN_STATUS}"
fi

echo "Finished ${RUN_NAME}"
