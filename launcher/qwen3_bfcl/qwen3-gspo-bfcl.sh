#!/bin/bash
# Single-node GSPO training on BFCL.
# 1) Set CONDA_SH (and optionally SWANLAB_API_KEY, BFCL_CONDA_ENV, TRAIN_CONDA_ENV, BFCL_ENV_DIR).
# 2) Run from anywhere:  bash /path/to/SeeUPO/launcher/qwen3_bfcl/qwen3-gspo-bfcl.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="${CONDA_SH:?Set CONDA_SH to your conda profile, e.g. \$HOME/miniconda3/etc/profile.d/conda.sh}"

BFCL_CONDA_ENV="${BFCL_CONDA_ENV:-bfcl}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-seeupo}"
export BFCL_ENV_DIR="${BFCL_ENV_DIR:-${PROJECT_ROOT}/env_service/environments/bfcl}"
LOG_FILE="${PROJECT_ROOT}/bfcl_service.log"

echo "[1/2] Starting BFCL env service (nohup) -> ${LOG_FILE}"
nohup bash -c "
    . \"${CONDA_SH}\"
    conda activate \"${BFCL_CONDA_ENV}\"
    cd \"${PROJECT_ROOT}/env_service/launch_script\"
    exec bash bfcl.sh
" > "${LOG_FILE}" 2>&1 &
echo "BFCL env service PID: $!"
sleep "${BFCL_STARTUP_SLEEP:-10}"

echo "[2/2] Starting training (launcher.py)..."
. "${CONDA_SH}"
conda activate "${TRAIN_CONDA_ENV}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?Set SWANLAB_API_KEY}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
exec python launcher.py --conf "${PROJECT_ROOT}/launcher/qwen3_bfcl/qwen3-gspo-bfcl.yaml"
