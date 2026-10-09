#!/usr/bin/env bash
# Foundation X BRAVE: submit training only (silence kept + RandomGain).
#
# Variants:
#   shotgun           — DB: …/preprocessed_shotgun
#   shotgun_contact   — DB: …/preprocessed_shotgun_contact  (build LMDB yourself first)
#   both              — submit both (default)
#
# Preprocess joint LMDB separately, e.g.:
#   INPUT_PATH=$BRAVE_STORAGE/tap_samples/audio_subset_shotgun \
#   INPUT_PATH_EXTRA=$BRAVE_STORAGE/tap_samples/audio_subset_clean \
#   OUTPUT_PATH=$BRAVE_STORAGE/tap_samples/preprocessed_shotgun_contact \
#   NORMALIZE=0 sbatch --chdir=$BRAVE_ROOT scripts/preprocess.sbatch
#
# Usage (login node, from repo root):
#   bash scripts/submit_foundation_x_train.sh
#   VARIANT=shotgun bash scripts/submit_foundation_x_train.sh
#   bash scripts/submit_foundation_x_train.sh --gres=gpu:2 --cpus-per-task=16
#
# Foundation Y: bash scripts/submit_foundation_y_batch.sh

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

VARIANT="${VARIANT:-both}"  # both | shotgun | shotgun_contact
CONFIG="${CONFIG:-configs/brave_tap.gin}"
OUT_PATH="${OUT_PATH:-${BRAVE_STORAGE}/tap_samples/runs}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"

DB_SHOTGUN="${DB_SHOTGUN:-${BRAVE_STORAGE}/tap_samples/preprocessed_shotgun}"
DB_SHOTGUN_CONTACT="${DB_SHOTGUN_CONTACT:-${BRAVE_STORAGE}/tap_samples/preprocessed_shotgun_contact}"

unset CKPT WANDB_RUN_ID GPUS GPU DB_PATH DB_PATH_X DB_PATH_Y DOMAIN_X_FRACTION
mkdir -p logs

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run on the cluster login node." >&2
  exit 1
fi

case "${VARIANT}" in
  both|shotgun|shotgun_contact) ;;
  *)
    echo "ERROR: VARIANT must be both|shotgun|shotgun_contact (got ${VARIANT})" >&2
    exit 1
    ;;
esac

cli_args=("$@")

submit_one() {
  local label="$1"
  shift
  local -a sbatch_extra=("$@")
  echo "Submitting Foundation X (${label})"
  echo "  RUN_NAME=${RUN_NAME}"
  echo "  CONFIG=${CONFIG}"
  echo "  OUT_PATH=${OUT_PATH}"
  echo "  DB_PATH=${DB_PATH}"
  local job_id
  job_id="$(
    sbatch --parsable \
      --chdir="${BRAVE_ROOT}" \
      --job-name="found-x-${label}" \
      --output="${BRAVE_ROOT}/logs/train-%j.log" \
      --error="${BRAVE_ROOT}/logs/train-%j.log" \
      "${sbatch_extra[@]}" \
      scripts/train.sbatch
  )" || {
    echo "ERROR: sbatch failed for ${label}" >&2
    exit 1
  }
  echo "  job ${job_id}  log: ${BRAVE_ROOT}/logs/train-${job_id}.log"
  echo ""
}

if [[ "${VARIANT}" == "both" || "${VARIANT}" == "shotgun" ]]; then
  if [[ ! -d "${DB_SHOTGUN}" ]]; then
    echo "ERROR: shotgun LMDB missing: ${DB_SHOTGUN}" >&2
    exit 1
  fi
  export DB_PATH="${DB_SHOTGUN}"
  export RUN_NAME="${RUN_NAME_SHOTGUN:-tap_foundation_x_shotgun}"
  export CONFIG OUT_PATH BATCH WORKERS
  submit_one shotgun "${cli_args[@]}"
fi

if [[ "${VARIANT}" == "both" || "${VARIANT}" == "shotgun_contact" ]]; then
  if [[ ! -d "${DB_SHOTGUN_CONTACT}" ]]; then
    echo "ERROR: joint shotgun+contact LMDB missing: ${DB_SHOTGUN_CONTACT}" >&2
    echo "Preprocess first (NORMALIZE=0), then re-run this script." >&2
    exit 1
  fi
  export DB_PATH="${DB_SHOTGUN_CONTACT}"
  export RUN_NAME="${RUN_NAME_CONTACT:-tap_foundation_x_shotgun_contact}"
  export CONFIG OUT_PATH BATCH WORKERS
  submit_one shotgun_contact "${cli_args[@]}"
fi

echo "Done. Watch: squeue -u \$USER"
