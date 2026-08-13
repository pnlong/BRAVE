#!/usr/bin/env bash
#SBATCH --job-name=brave-cyclegan
#SBATCH --output=logs/train-cyclegan-%j.log
#SBATCH --error=logs/train-cyclegan-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

# Dual frozen BRAVE CycleGAN (latent warps; waveform warps shelved).
# Domain X = source (e.g. tap), domain Y = target (e.g. birdsong / water).
# CYCLE_DOMAIN=waveform|latent (optional gin override). LATENT_CYCLE_MODE=ae_aware|direct.
: "${CKPT_X:?Set CKPT_X to domain-X checkpoint or run directory}"
: "${DB_PATH_X:?Set DB_PATH_X to domain-X LMDB}"
: "${CKPT_Y:?Set CKPT_Y to domain-Y checkpoint or run directory}"
: "${DB_PATH_Y:?Set DB_PATH_Y to domain-Y LMDB}"
: "${CANONICALIZER_TYPE:=latent}"
: "${RUN_NAME:?Set RUN_NAME for output folder name}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"

BACKBONE_X_CONFIG="${BACKBONE_X_CONFIG:-configs/brave.gin}"
BACKBONE_Y_CONFIG="${BACKBONE_Y_CONFIG:-configs/brave.gin}"
CYCLEGAN_CONFIG="${CYCLEGAN_CONFIG:-configs/brave_cyclegan.gin}"
OUT_PATH="${OUT_PATH:-${BRAVE_RUNS:-${BRAVE_ROOT}/runs}}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
N_SIGNAL="${N_SIGNAL:-131072}"
MAX_STEPS="${MAX_STEPS:-}"
VAL_EVERY="${VAL_EVERY:-500}"
WANDB_PROJECT="${WANDB_PROJECT:-brave}"

mkdir -p logs "${OUT_PATH}"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

# Multi-GPU: sbatch --gres=gpu:2 --cpus-per-task=16 ...
# Optional explicit ids: GPUS=0,1  (default: 0..N-1 from SLURM allocation)
GPU_TRAIN_ARGS=()
slurm_build_gpu_train_args GPU_TRAIN_ARGS

cd "${BRAVE_ROOT}"

# Avoid leaking backbone train.sbatch CONFIG into this job.
unset CONFIG

echo "CYCLEGAN_CONFIG=${CYCLEGAN_CONFIG}"
echo "BACKBONE_X_CONFIG=${BACKBONE_X_CONFIG}"
echo "CKPT_X=${CKPT_X}"
echo "DB_PATH_X=${DB_PATH_X}"
echo "BACKBONE_Y_CONFIG=${BACKBONE_Y_CONFIG}"
echo "CKPT_Y=${CKPT_Y}"
echo "DB_PATH_Y=${DB_PATH_Y}"
echo "CANONICALIZER_TYPE=${CANONICALIZER_TYPE}"
echo "CYCLE_DOMAIN=${CYCLE_DOMAIN:-}"
echo "LATENT_CYCLE_MODE=${LATENT_CYCLE_MODE:-}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUT_PATH=${OUT_PATH}"
if [[ -n "${OVERRIDE:-}" ]]; then
  echo "OVERRIDE=${OVERRIDE}"
fi
if [[ -n "${MAX_STEPS}" ]]; then
  echo "MAX_STEPS=${MAX_STEPS} (CLI override)"
else
  echo "MAX_STEPS=(from CYCLE_MAX_STEPS in ${CYCLEGAN_CONFIG})"
fi
echo "GPUs: ${GPU_TRAIN_ARGS[*]}"

CYCLE_ARGS=(
  --config="${CYCLEGAN_CONFIG}"
  --backbone_x_config="${BACKBONE_X_CONFIG}"
  --ckpt_x="${CKPT_X}"
  --db_path_x="${DB_PATH_X}"
  --backbone_y_config="${BACKBONE_Y_CONFIG}"
  --ckpt_y="${CKPT_Y}"
  --db_path_y="${DB_PATH_Y}"
  --canonicalizer_type="${CANONICALIZER_TYPE}"
  --name="${RUN_NAME}"
  --out_path="${OUT_PATH}"
  --batch="${BATCH}"
  --workers="${WORKERS}"
  --n_signal="${N_SIGNAL}"
  --val_every="${VAL_EVERY}"
  --wandb_project="${WANDB_PROJECT}"
  "${GPU_TRAIN_ARGS[@]}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  CYCLE_ARGS+=(--max_steps="${MAX_STEPS}")
fi

if [[ "${CANONICALIZER_TYPE}" == "waveform" ]]; then
  echo "WARNING: waveform CycleGAN warps are shelved; prefer CANONICALIZER_TYPE=latent."
fi

# Optional domain switches (appended before OVERRIDE).
if [[ -n "${CYCLE_DOMAIN:-}" ]]; then
  CYCLE_ARGS+=(--override="CycleGANTrainer.cycle_domain=\"${CYCLE_DOMAIN}\"")
fi
if [[ -n "${LATENT_CYCLE_MODE:-}" ]]; then
  CYCLE_ARGS+=(--override="CycleGANTrainer.latent_cycle_mode=\"${LATENT_CYCLE_MODE}\"")
fi

# Gin overrides: OVERRIDE='binding1 binding2' (space-separated).
if [[ -n "${OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  for ov in ${OVERRIDE}; do
    CYCLE_ARGS+=(--override="${ov}")
  done
fi

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  CYCLE_ARGS+=(--smoke_test)
fi

python RAVE/scripts/train_cyclegan.py "${CYCLE_ARGS[@]}"
