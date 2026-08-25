#!/usr/bin/env bash
# After tap hits epoch_1500000.ckpt, submit wf-cyc-mlp2 CycleGAN for each
# yt_playlist backbone.
#
# 1) Resume tap (needs the global_step checkpoint fix in rave.core.ModelCheckpoint):
#    CKPT=/data/scratch-fast/p1long/BRAVE/tap_samples/runs/tap_uncond_run_8e1e614287 \
#    DB_PATH=/data/scratch-fast/p1long/BRAVE/tap_samples/preprocessed_normalized_clean \
#    RUN_NAME=tap_uncond_run CONFIG=configs/brave.gin \
#    OUT_PATH=/data/scratch-fast/p1long/BRAVE/tap_samples/runs \
#    WANDB_RUN_ID=zx8uo6l3 MAX_STEPS=1500000 \
#    sbatch --chdir=/data/hai-res/p1long/BRAVE --job-name=tap-uncond-brave \
#      --output=/data/hai-res/p1long/BRAVE/logs/train-%j.log \
#      --error=/data/hai-res/p1long/BRAVE/logs/train-%j.log \
#      --gres=gpu:4 --cpus-per-task=32 scripts/train.sbatch
#
# 2) Then, from a login node:
#      sleep 2h; bash scripts/deploy_cyclegan_playlists.sh
#    The script polls for tap *and* each playlist epoch_1500000.ckpt
#    (including birds_chirping), so a short sleep is OK.
#
# FRESH=1 so we do not resume the Aug 16 CycleGAN last.ckpt dirs.

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
SCRATCH="${SCRATCH:-/data/scratch-fast/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

TAP_RUN="${TAP_RUN:-${SCRATCH}/tap_samples/runs/tap_uncond_run_8e1e614287}"
CKPT_X="${CKPT_X:-${TAP_RUN}/epoch_1500000.ckpt}"
DB_PATH_X="${DB_PATH_X:-${SCRATCH}/tap_samples/preprocessed_normalized_clean}"
PLAYLISTS_ROOT="${PLAYLISTS_ROOT:-${SCRATCH}/yt_playlists}"

WAIT_SECS="${WAIT_SECS:-21600}"   # 6h
POLL_SECS="${POLL_SECS:-60}"

PLAYLISTS=(
  rain_sounds
  babbling_brook
  waves_crashing
  fire_crackling
  night_ambience
  birds_chirping
)

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the cluster login node." >&2
  exit 1
fi

wait_for_ckpt() {
  local path="$1"
  echo "Waiting for ${path}"
  local elapsed=0
  while [[ ! -f "${path}" ]]; do
    if (( elapsed >= WAIT_SECS )); then
      echo "ERROR: still no ${path} after ${WAIT_SECS}s" >&2
      exit 1
    fi
    echo "  not yet (${elapsed}s / ${WAIT_SECS}s); sleep ${POLL_SECS}s"
    sleep "${POLL_SECS}"
    elapsed=$((elapsed + POLL_SECS))
  done
  echo "  found $(ls -lh "${path}" | awk '{print $5, $6, $7, $8}')"
}

wait_for_ckpt "${CKPT_X}"
for p in "${PLAYLISTS[@]}"; do
  wait_for_ckpt "${PLAYLISTS_ROOT}/${p}/runs/${p}_run_8e1e614287/epoch_1500000.ckpt"
done

mkdir -p logs

submit_one() {
  local playlist="$1"
  local ckpt_y="${PLAYLISTS_ROOT}/${playlist}/runs/${playlist}_run_8e1e614287/epoch_1500000.ckpt"

  # Subshell so tap WANDB_RUN_ID / CKPT / MAX_STEPS cannot leak into CycleGAN.
  local job_id
  job_id="$(
    unset CKPT WANDB_RUN_ID MAX_STEPS GPUS GPU RESUME OVERRIDE CYCLE_DOMAIN LATENT_CYCLE_MODE
    export CKPT_X CKPT_Y="${ckpt_y}" \
      DB_PATH_X \
      DB_PATH_Y="${PLAYLISTS_ROOT}/${playlist}/preprocessed" \
      BACKBONE_X_CONFIG=configs/brave.gin \
      BACKBONE_Y_CONFIG=configs/brave.gin \
      CYCLEGAN_CONFIG=configs/brave_cyclegan_best.gin \
      CANONICALIZER_TYPE=latent \
      FRESH=1 \
      RUN_NAME="tap_${playlist}_wf_cyc_mlp2" \
      OUT_PATH="${PLAYLISTS_ROOT}/${playlist}/runs"
    sbatch --parsable \
      --chdir="${BRAVE_ROOT}" \
      --job-name="cyc-${playlist}" \
      --output="${BRAVE_ROOT}/logs/train-cyclegan-%j.log" \
      --error="${BRAVE_ROOT}/logs/train-cyclegan-%j.log" \
      scripts/train_cyclegan.sbatch
  )" || {
    echo "ERROR: sbatch failed for ${playlist}" >&2
    return 1
  }
  echo "  ${playlist}: job ${job_id}  log=logs/train-cyclegan-${job_id}.log"
}

echo "Submitting CycleGAN jobs (FRESH=1, brave_cyclegan_best.gin)..."
for p in "${PLAYLISTS[@]}"; do
  submit_one "${p}"
done
echo "Done."
