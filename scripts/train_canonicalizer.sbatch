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

: "${FADER_CONFIG:?Set FADER_CONFIG (e.g. configs/brave_fader_birdsong.gin)}"
: "${CKPT:?Set CKPT to frozen FaderRAVE checkpoint or run directory}"
: "${DB_PATH:?Set DB_PATH to in-domain Fader LMDB}"
: "${OOD_DB_PATH:?Set OOD_DB_PATH to out-of-domain Fader LMDB}"
: "${CANONICALIZER_TYPE:?Set CANONICALIZER_TYPE to waveform or latent}"
: "${RUN_NAME:?Set RUN_NAME for output folder name}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"

CONFIG="${CONFIG:-configs/brave_canonicalizer.gin}"
OUT_PATH="${OUT_PATH:-${BRAVE_RUNS:-${BRAVE_ROOT}/runs}}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
N_SIGNAL="${N_SIGNAL:-131072}"
MAX_STEPS="${MAX_STEPS:-10000}"
VAL_EVERY="${VAL_EVERY:-500}"
GPU="${GPU:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-brave}"
IR_PATH="${IR_PATH:-}"
IR_PROB="${IR_PROB:-}"

mkdir -p logs "${OUT_PATH}"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

cd "${BRAVE_ROOT}"

echo "FADER_CONFIG=${FADER_CONFIG}"
echo "CKPT=${CKPT}"
echo "DB_PATH=${DB_PATH}"
echo "OOD_DB_PATH=${OOD_DB_PATH}"
echo "CANONICALIZER_TYPE=${CANONICALIZER_TYPE}"
echo "RUN_NAME=${RUN_NAME}"

CANON_ARGS=(
  --config="${CONFIG}"
  --fader_config="${FADER_CONFIG}"
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
  --gpu="${GPU}"
  --wandb_project="${WANDB_PROJECT}"
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
