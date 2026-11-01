#!/usr/bin/env bash
# Submit stratified joint Fader foundation (tap ∪ water, schema x∪y).
#
# Usage (login node, from repo root):
#   bash scripts/submit_joint_fader_train.sh
#   DB_PATH_X=... DB_PATH_Y=... bash scripts/submit_joint_fader_train.sh

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

export DB_PATH_X="${DB_PATH_X:-${BRAVE_STORAGE}/tap_samples/preprocessed_normalized_clean}"
export DB_PATH_Y="${DB_PATH_Y:-${BRAVE_STORAGE}/fsd50k_water/preprocessed}"
export DOMAIN_X_FRACTION="${DOMAIN_X_FRACTION:-0.5}"
export RUN_NAME="${RUN_NAME:-joint_fader_tap_water}"
export CONFIG="${CONFIG:-configs/brave_fader_joint_tap_water.gin}"
export OUT_PATH="${OUT_PATH:-${BRAVE_STORAGE}/joint_fader_tap_water/runs}"
export BATCH="${BATCH:-8}"
export WORKERS="${WORKERS:-8}"

unset CKPT WANDB_RUN_ID GPUS GPU DB_PATH

mkdir -p logs

echo "Submitting stratified joint Fader:"
echo "  DB_PATH_X=${DB_PATH_X}"
echo "  DB_PATH_Y=${DB_PATH_Y}"
echo "  DOMAIN_X_FRACTION=${DOMAIN_X_FRACTION}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  CONFIG=${CONFIG}"
echo "  OUT_PATH=${OUT_PATH}"
echo ""
echo "Ensure attribute_stats.yaml exists on both LMDBs first:"
echo "  python RAVE/scripts/precompute_descriptors.py --db_path=\$DB_PATH_X --config=\$CONFIG --train_only"
echo "  python RAVE/scripts/precompute_descriptors.py --db_path=\$DB_PATH_Y --config=\$CONFIG --train_only"
echo "Water LMDB needs attribute_sidecar.yaml with water_scene (tap may omit it)."

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run on the cluster login node." >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    --chdir="${BRAVE_ROOT}" \
    --job-name=joint-fader \
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
