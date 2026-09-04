#!/usr/bin/env bash
# Submit stratified joint AE training (contact tap + yt_birdsong, 50/50 batches).
#
# Uses two separate LMDBs so sampling can balance domains.
#
# Usage (login node, from repo root):
#   bash scripts/submit_joint_train.sh
#   bash scripts/submit_joint_train.sh --gres=gpu:4 --cpus-per-task=32 --mem=128G

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

# Contact (normalized clean) + birdsong (PCEN+normalize — prior birdsong BRAVE LMDB).
export DB_PATH_X="${DB_PATH_X:-${DB_PATH:-${BRAVE_STORAGE}/tap_samples/preprocessed_normalized_clean}}"
export DB_PATH_Y="${DB_PATH_Y:-${BRAVE_STORAGE}/yt_birdsong/preprocessed_pcen}"
export DOMAIN_X_FRACTION="${DOMAIN_X_FRACTION:-0.5}"
export RUN_NAME="${RUN_NAME:-joint_tap_contact_birdsong_balanced}"
export CONFIG="${CONFIG:-configs/brave_birdsong.gin}"
export OUT_PATH="${OUT_PATH:-${BRAVE_STORAGE}/joint_tap_contact_birdsong/runs}"
export BATCH="${BATCH:-8}"
export WORKERS="${WORKERS:-8}"

unset CKPT WANDB_RUN_ID GPUS GPU DB_PATH

mkdir -p logs

echo "Submitting stratified joint BRAVE:"
echo "  DB_PATH_X=${DB_PATH_X}"
echo "  DB_PATH_Y=${DB_PATH_Y}"
echo "  DOMAIN_X_FRACTION=${DOMAIN_X_FRACTION}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  CONFIG=${CONFIG}"
echo "  OUT_PATH=${OUT_PATH}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run on the cluster login node." >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    --chdir="${BRAVE_ROOT}" \
    --job-name=joint-brave \
    --output="${BRAVE_ROOT}/logs/train-%j.log" \
    --error="${BRAVE_ROOT}/logs/train-%j.log" \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    --mem=128G \
    "$@" \
    scripts/train.sbatch
)" || {
  echo "ERROR: sbatch failed." >&2
  exit 1
}

echo ""
echo "Submitted job ${job_id}"
echo "Log: ${BRAVE_ROOT}/logs/train-${job_id}.log"
echo "Watch: tail -f ${BRAVE_ROOT}/logs/train-${job_id}.log"
