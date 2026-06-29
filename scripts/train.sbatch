#!/usr/bin/env bash
#SBATCH --job-name=brave-train
#SBATCH --output=logs/train-%j.log
#SBATCH --error=logs/train-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

: "${DB_PATH:?Set DB_PATH to preprocessed LMDB directory}"
: "${RUN_NAME:?Set RUN_NAME to wandb/run folder name}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"

CONFIG="${CONFIG:-configs/brave.gin}"
OUT_PATH="${OUT_PATH:-${BRAVE_RUNS:-${BRAVE_ROOT}/runs}}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
CHANNELS="${CHANNELS:-1}"
MAX_STEPS="${MAX_STEPS:-1500000}"
WANDB_PROJECT="${WANDB_PROJECT:-brave}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
CKPT="${CKPT:-}"

# Multi-GPU: request more GPUs when submitting, e.g.
#   sbatch --gres=gpu:4 --cpus-per-task=32 scripts/train.sbatch
# Optional explicit ids: GPUS=0,1,2,3  (default: 0..N-1 from SLURM allocation)
# Per-GPU batch size is BATCH; global batch ≈ BATCH × num_gpus.

mkdir -p logs "${OUT_PATH}"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

GPU_TRAIN_ARGS=()
slurm_build_gpu_train_args GPU_TRAIN_ARGS

cd "${BRAVE_ROOT}"

echo "DB_PATH=${DB_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "CONFIG=${CONFIG}"
echo "GPUs: ${GPU_TRAIN_ARGS[*]}"
if [[ -n "${CKPT}" ]]; then
  echo "CKPT=${CKPT}"
fi
python scripts/lmdb_duration.py "${DB_PATH}"

TRAIN_ARGS=(
  --config="${CONFIG}"
  --name="${RUN_NAME}"
  --db_path="${DB_PATH}"
  --out_path="${OUT_PATH}"
  --batch="${BATCH}"
  --workers="${WORKERS}"
  --channels="${CHANNELS}"
  --max_steps="${MAX_STEPS}"
  --wandb_project="${WANDB_PROJECT}"
  "${GPU_TRAIN_ARGS[@]}"
)
if [[ -n "${WANDB_RUN_ID}" ]]; then
  TRAIN_ARGS+=(--wandb_run_id="${WANDB_RUN_ID}")
fi
if [[ -n "${CKPT}" ]]; then
  TRAIN_ARGS+=(--ckpt="${CKPT}")
fi

python RAVE/scripts/train.py "${TRAIN_ARGS[@]}"
