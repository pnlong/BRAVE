#!/usr/bin/env bash
#SBATCH --job-name=brave-canon
#SBATCH --output=logs/train-canonicalizer-%j.log
#SBATCH --error=logs/train-canonicalizer-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

# Frozen RAVE or FaderRAVE architecture gin (e.g. configs/brave_birdsong.gin).
: "${BACKBONE_CONFIG:?Set BACKBONE_CONFIG (e.g. configs/brave_birdsong.gin or configs/brave_fader_birdsong.gin)}"
: "${CKPT:?Set CKPT to frozen RAVE / FaderRAVE checkpoint or run directory}"
: "${DB_PATH:?Set DB_PATH to in-domain LMDB}"
: "${OOD_DB_PATH:?Set OOD_DB_PATH to out-of-domain LMDB}"
: "${CANONICALIZER_TYPE:?Set CANONICALIZER_TYPE to waveform or latent}"
: "${RUN_NAME:?Set RUN_NAME for output folder name}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"

# Do not use CONFIG here — train.sbatch sets CONFIG to the backbone gin, and
# sbatch --export=ALL would leak that into canonicalizer runs. Use a dedicated var.
CANONICALIZER_CONFIG="${CANONICALIZER_CONFIG:-configs/brave_canonicalizer.gin}"
OUT_PATH="${OUT_PATH:-${BRAVE_RUNS:-${BRAVE_ROOT}/runs}}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
N_SIGNAL="${N_SIGNAL:-131072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_EVERY="${VAL_EVERY:-500}"
WANDB_PROJECT="${WANDB_PROJECT:-brave}"
IR_PATH="${IR_PATH:-}"
IR_PROB="${IR_PROB:-}"

mkdir -p logs "${OUT_PATH}"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

# Multi-GPU: sbatch --gres=gpu:2 --cpus-per-task=16 ...
# Optional explicit ids: GPUS=0,1  (default: 0..N-1 from SLURM allocation)
GPU_TRAIN_ARGS=()
slurm_build_gpu_train_args GPU_TRAIN_ARGS

cd "${BRAVE_ROOT}"

# CONFIG is commonly set for backbone train.sbatch; do not let it leak here.
unset CONFIG

echo "BACKBONE_CONFIG=${BACKBONE_CONFIG}"
echo "CANONICALIZER_CONFIG=${CANONICALIZER_CONFIG}"
echo "CKPT=${CKPT}"
echo "DB_PATH=${DB_PATH}"
echo "OOD_DB_PATH=${OOD_DB_PATH}"
echo "CANONICALIZER_TYPE=${CANONICALIZER_TYPE}"
echo "RUN_NAME=${RUN_NAME}"
echo "GPUs: ${GPU_TRAIN_ARGS[*]}"

CANON_ARGS=(
  --config="${CANONICALIZER_CONFIG}"
  --backbone_config="${BACKBONE_CONFIG}"
  --ckpt="${CKPT}"
  --db_path="${DB_PATH}"
  --ood_db_path="${OOD_DB_PATH}"
  --canonicalizer_type="${CANONICALIZER_TYPE}"
  --name="${RUN_NAME}"
  --out_path="${OUT_PATH}"
  --batch="${BATCH}"
  --workers="${WORKERS}"
  --n_signal="${N_SIGNAL}"
  --max_steps="${MAX_STEPS}"
  --val_every="${VAL_EVERY}"
  --wandb_project="${WANDB_PROJECT}"
  "${GPU_TRAIN_ARGS[@]}"
)

if [[ -n "${IR_PATH}" ]]; then
  CANON_ARGS+=(--ir_path="${IR_PATH}")
fi
if [[ -n "${IR_PROB}" ]]; then
  CANON_ARGS+=(--ir_prob="${IR_PROB}")
fi
if [[ "${NO_IR:-0}" == "1" ]]; then
  CANON_ARGS+=(--no_ir)
fi
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  CANON_ARGS+=(--smoke_test)
fi

python RAVE/scripts/train_canonicalizer.py "${CANON_ARGS[@]}"
