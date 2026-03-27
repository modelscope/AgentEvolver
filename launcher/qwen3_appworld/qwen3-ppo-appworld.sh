#!/bin/bash
# Multi-node PPO training on AppWorld (Ray cluster).
# Rank 0: starts AppWorld env service (via appworld.sh), Ray head, then training. Other ranks: Ray workers.
#
# Required (export before running, or set in your job wrapper):
#   CONDA_SH          Path to conda.sh (e.g. $HOME/miniconda3/etc/profile.d/conda.sh)
#   SWANLAB_API_KEY   Logging API key
# Cluster / scheduler (typical for DLC, Slurm, etc.):
#   RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#
# Optional:
#   APPWORLD_CONDA_ENV (default: appworld), TRAIN_CONDA_ENV (default: seeupo)
#   APPWORLD_ROOT     If set, exported before starting appworld.sh (AppWorld assets location)
#   NUM_GPUS_PER_NODE (default: 8), NUM_CPUS_PER_NODE (default: 64)
#   OBJECT_STORE_MEMORY (default: 100000000000), RAY_HEAD_PORT (default: 6379)
#   APPWORLD_SERVICE_URL (informational; training YAML should match appworld.sh port, default 8080)
#   NCCL_* / GLOO_*    Tune for your network

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="${CONDA_SH:?Set CONDA_SH to your conda profile (conda.sh path)}"

export SWANLAB_API_KEY="${SWANLAB_API_KEY:?Set SWANLAB_API_KEY}"
export APPWORLD_SERVICE_URL="${APPWORLD_SERVICE_URL:-http://localhost:8080}"
export RAY_HEAD_NODE="${MASTER_ADDR:?MASTER_ADDR not set}"
export RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

export PROJECT_PATH="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_PATH}:${PYTHONPATH:-}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
export NCCL_NTHREADS="${NCCL_NTHREADS:-4}"
export GLOO_DEVICE_TRANSPORT="${GLOO_DEVICE_TRANSPORT:-TCP}"
export GLOO_TIMEOUT="${GLOO_TIMEOUT:-7200}"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:-64}"
OBJECT_STORE_MEMORY="${OBJECT_STORE_MEMORY:-100000000000}"

APPWORLD_CONDA_ENV="${APPWORLD_CONDA_ENV:-appworld}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-seeupo}"
LOG_DIR="${PROJECT_ROOT}/logs/appworld"

print_green() { echo -e "\033[32m$1\033[0m"; }
print_red() { echo -e "\033[31m$1\033[0m"; }
print_yellow() { echo -e "\033[33m$1\033[0m"; }

print_green "==================================="
print_green "Node Rank: ${RANK}"
print_green "World Size: ${WORLD_SIZE}"
print_green "Master: ${MASTER_ADDR}:${MASTER_PORT}"
print_green "==================================="

if [ "${RANK}" = "0" ]; then
    print_green "=========================================="
    print_green "MAIN NODE (Rank 0) - Starting services..."
    print_green "=========================================="

    print_green "[1/4] Starting AppWorld env service in background..."
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/appworld_service_$(date +%Y%m%d_%H%M%S).log"

    . "${CONDA_SH}"
    conda activate "${APPWORLD_CONDA_ENV}"

    # Optional: export APPWORLD_ROOT before running if your layout differs from the default under env_service/environments/appworld
    if [ -n "${APPWORLD_ROOT:-}" ]; then
        export APPWORLD_ROOT
    fi

    cd "${PROJECT_ROOT}/env_service/launch_script"
    nohup bash appworld.sh > "${LOG_FILE}" 2>&1 &
    APPWORLD_PID=$!

    print_green "AppWorld service started with PID: ${APPWORLD_PID}"
    print_green "AppWorld logs: ${LOG_FILE}"

    print_yellow "Waiting for AppWorld service to initialize..."
    sleep 15

    print_green "[2/4] Switching to training environment..."
    conda activate "${TRAIN_CONDA_ENV}" || { print_red "Failed to activate ${TRAIN_CONDA_ENV}"; exit 1; }

    print_green "[3/4] Starting Ray head node..."
    ray start --head \
        --port="${RAY_HEAD_PORT}" \
        --num-gpus="${NUM_GPUS_PER_NODE}" \
        --num-cpus="${NUM_CPUS_PER_NODE}" \
        --object-store-memory="${OBJECT_STORE_MEMORY}" \
        --dashboard-host=0.0.0.0

    print_green "Ray head started successfully"
    sleep 10

    print_yellow "Waiting for worker nodes to join..."
    sleep 10

    print_green "=========================================="
    ray status
    print_green "=========================================="

    export RAY_ADDRESS="ray://localhost:10001"

    print_green "[4/4] Starting training..."
    cd "${PROJECT_ROOT}"
    python launcher_multinode.py \
        --conf "${PROJECT_ROOT}/launcher/qwen3_appworld/qwen3-ppo-appworld.yaml"

    TRAINING_EXIT_CODE=$?

    print_green "=========================================="
    print_green "Training finished with exit code: ${TRAINING_EXIT_CODE}"
    print_green "Cleaning up services..."

    ray stop

    if kill -0 "${APPWORLD_PID}" 2>/dev/null; then
        print_yellow "Stopping AppWorld service (PID: ${APPWORLD_PID})..."
        kill "${APPWORLD_PID}"
        sleep 3
        kill -9 "${APPWORLD_PID}" 2>/dev/null || true
    fi

    print_green "Cleanup completed"
    exit "${TRAINING_EXIT_CODE}"

else
    print_green "=========================================="
    print_green "WORKER NODE (Rank ${RANK}) - Joining cluster..."
    print_green "=========================================="

    . "${CONDA_SH}"
    conda activate "${TRAIN_CONDA_ENV}" || { print_red "Failed to activate ${TRAIN_CONDA_ENV}"; exit 1; }

    export PYTHONPATH="${PROJECT_PATH}:${PYTHONPATH:-}"

    print_yellow "Waiting for Ray head node to be ready..."
    sleep 25

    print_green "Starting Ray worker node..."
    ray start \
        --address="${RAY_HEAD_NODE}:${RAY_HEAD_PORT}" \
        --num-gpus="${NUM_GPUS_PER_NODE}" \
        --num-cpus="${NUM_CPUS_PER_NODE}" \
        --object-store-memory="${OBJECT_STORE_MEMORY}"

    print_green "Ray worker started successfully"
    sleep 5

    print_green "=========================================="
    ray status
    print_green "=========================================="

    print_green "Worker node ready, waiting for training to complete..."
    tail -f /dev/null
fi
