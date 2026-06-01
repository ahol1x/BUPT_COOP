#!/usr/bin/env bash
set -e
set -o pipefail

REPO_DIR="/home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT"
CONDA_SH="/home/kennycao12/miniconda3/etc/profile.d/conda.sh"
EXPECTED_PYTHON_PREFIX="/home/kennycao12/miniconda3/envs/bupt-cil/"

cd "${REPO_DIR}"

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

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
if [ "${CONDA_DEFAULT_ENV:-}" != "bupt-cil" ]; then
    echo "ERROR: expected conda env bupt-cil."
    exit 1
fi

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

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU name from nvidia-smi:"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

python - <<'PY'
import importlib
import sys

modules = [
    "torch",
    "torchvision",
    "timm",
    "tqdm",
    "numpy",
    "scipy",
    "PIL",
    "easydict",
]

failed = []
for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "version unavailable")
        print(f"import {name}: OK ({version})")
    except Exception as exc:
        print(f"import {name}: FAIL ({exc})")
        failed.append(name)

if failed:
    print("Missing or broken modules: " + ", ".join(failed))
    sys.exit(1)

import torch

print("torch version:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("torch gpu 0:", torch.cuda.get_device_name(0))
else:
    sys.exit(1)
PY

echo "Checking python main.py --help:"
python main.py --help

echo "Step 0 server environment check passed."
