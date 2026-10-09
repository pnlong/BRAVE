#!/usr/bin/env bash
# CycleGAN: Foundation X (shotgun+contact) ↔ finished Foundation Y playlists.
#
# Recipe: configs/brave_cyclegan_best.gin
#   Approach 2 separate AEs, latent 2-layer MLP warps, waveform cycle + audio D,
#   400k steps (same as tap_*_wf_cyc_mlp2).
#
# Usage (login node, from repo root):
#   bash scripts/submit_foundation_cyclegan.sh
#   PLAYLISTS="babbling_brook waves_crashing" bash scripts/submit_foundation_cyclegan.sh
#
# Training only — LMDBs and epoch_1500000.ckpt must already exist.

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

CKPT_X="${CKPT_X:-${BRAVE_STORAGE}/tap_samples/runs/tap_foundation_x_shotgun_contact_70d386ede1/epoch_1500000.ckpt}"
DB_PATH_X="${DB_PATH_X:-${BRAVE_STORAGE}/tap_samples/preprocessed_shotgun_contact}"
PLAYLISTS_ROOT="${PLAYLISTS_ROOT:-${BRAVE_STORAGE}/yt_playlists}"

DEFAULT_PLAYLISTS=(babbling_brook waves_crashing night_ambience)
if [[ -n "${PLAYLISTS:-}" ]]; then
  # shellcheck disable=SC2206
  SELECTED=(${PLAYLISTS})
else
  SELECTED=("${DEFAULT_PLAYLISTS[@]}")
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run on the cluster login node." >&2
  exit 1
fi

if [[ ! -f "${CKPT_X}" ]]; then
  echo "ERROR: Foundation X ckpt missing: ${CKPT_X}" >&2
  exit 1
fi
if [[ ! -d "${DB_PATH_X}" ]]; then
  echo "ERROR: Foundation X LMDB missing: ${DB_PATH_X}" >&2
  exit 1
fi

mkdir -p logs

echo "Foundation CycleGAN (brave_cyclegan_best.gin)"
echo "  CKPT_X=${CKPT_X}"
echo "  DB_PATH_X=${DB_PATH_X}"
echo ""

for playlist in "${SELECTED[@]}"; do
  ckpt_y="${PLAYLISTS_ROOT}/${playlist}/runs/${playlist}_foundation_y_70d386ede1/epoch_1500000.ckpt"
  db_y="${PLAYLISTS_ROOT}/${playlist}/preprocessed"
  if [[ ! -f "${ckpt_y}" ]]; then
    echo "SKIP ${playlist}: missing ${ckpt_y}" >&2
    continue
  fi
  if [[ ! -d "${db_y}" ]]; then
    echo "SKIP ${playlist}: missing ${db_y}" >&2
    continue
  fi

  run_name="tap_fx_sc_${playlist}_wf_cyc_mlp2"
  job_id="$(
    unset CKPT WANDB_RUN_ID MAX_STEPS GPUS GPU RESUME OVERRIDE CYCLE_DOMAIN LATENT_CYCLE_MODE LAMBDA_LATENT_GAN AUDIO_POLISH_START
    export CKPT_X \
      CKPT_Y="${ckpt_y}" \
      DB_PATH_X \
      DB_PATH_Y="${db_y}" \
      BACKBONE_X_CONFIG=configs/brave.gin \
      BACKBONE_Y_CONFIG=configs/brave.gin \
      CYCLEGAN_CONFIG=configs/brave_cyclegan_best.gin \
      CANONICALIZER_TYPE=latent \
      FRESH=1 \
      RUN_NAME="${run_name}" \
      OUT_PATH="${PLAYLISTS_ROOT}/${playlist}/runs"
    sbatch --parsable \
      --chdir="${BRAVE_ROOT}" \
      --job-name="cyc-${playlist}" \
      --output="${BRAVE_ROOT}/logs/train-cyclegan-%j.log" \
      --error="${BRAVE_ROOT}/logs/train-cyclegan-%j.log" \
      "$@" \
      scripts/train_cyclegan.sbatch
  )" || {
    echo "ERROR: sbatch failed for ${playlist}" >&2
    exit 1
  }
  echo "  ${playlist}: job ${job_id}  RUN_NAME=${run_name}"
  echo "    log: ${BRAVE_ROOT}/logs/train-cyclegan-${job_id}.log"
done

echo ""
echo "Done. Watch: squeue -u \$USER"
echo "Each job: 1 GPU / 8 CPUs / 48G / up to 10h (resubmit same env to continue from last.ckpt)."
