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
GPU="${GPU:-0}"
MAX_STEPS="${MAX_STEPS:-1500000}"
WANDB_PROJECT="${WANDB_PROJECT:-brave}"

mkdir -p logs "${OUT_PATH}"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

cd "${BRAVE_ROOT}"

echo "DB_PATH=${DB_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "CONFIG=${CONFIG}"
python scripts/lmdb_duration.py "${DB_PATH}"

python RAVE/scripts/train.py \
  --config="${CONFIG}" \
  --name="${RUN_NAME}" \
  --db_path="${DB_PATH}" \
  --out_path="${OUT_PATH}" \
  --batch="${BATCH}" \
  --workers="${WORKERS}" \
  --channels="${CHANNELS}" \
  --gpu="${GPU}" \
  --max_steps="${MAX_STEPS}" \
  --wandb_project="${WANDB_PROJECT}"
