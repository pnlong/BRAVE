#!/usr/bin/env bash
# Foundation Y BRAVE: unconditional target-domain AEs (yt_playlists).
# Silence kept + RandomGain (from configs/brave.gin).
#
# Usage (login node, from repo root):
#   bash scripts/submit_foundation_y_batch.sh
#   DOMAINS="rain_sounds waves_crashing" bash scripts/submit_foundation_y_batch.sh
#
# Foundation X (tap): bash scripts/submit_foundation_x_train.sh

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

CONFIG="${CONFIG:-configs/brave.gin}"
RUN_SUFFIX="${RUN_SUFFIX:-foundation_y}"

# name|db_path|out_path
DEFAULT_DOMAINS=(
  "rain_sounds|${BRAVE_STORAGE}/yt_playlists/rain_sounds/preprocessed|${BRAVE_STORAGE}/yt_playlists/rain_sounds/runs"
  "waves_crashing|${BRAVE_STORAGE}/yt_playlists/waves_crashing/preprocessed|${BRAVE_STORAGE}/yt_playlists/waves_crashing/runs"
  "birds_chirping|${BRAVE_STORAGE}/yt_playlists/birds_chirping/preprocessed|${BRAVE_STORAGE}/yt_playlists/birds_chirping/runs"
  "babbling_brook|${BRAVE_STORAGE}/yt_playlists/babbling_brook/preprocessed|${BRAVE_STORAGE}/yt_playlists/babbling_brook/runs"
  "fire_crackling|${BRAVE_STORAGE}/yt_playlists/fire_crackling/preprocessed|${BRAVE_STORAGE}/yt_playlists/fire_crackling/runs"
  "night_ambience|${BRAVE_STORAGE}/yt_playlists/night_ambience/preprocessed|${BRAVE_STORAGE}/yt_playlists/night_ambience/runs"
)

if [[ -n "${DOMAINS:-}" ]]; then
  # shellcheck disable=SC2206
  SELECTED=(${DOMAINS})
  ENTRIES=()
  for name in "${SELECTED[@]}"; do
    found=
    for entry in "${DEFAULT_DOMAINS[@]}"; do
      if [[ "${entry%%|*}" == "${name}" ]]; then
        ENTRIES+=("${entry}")
        found=1
        break
      fi
    done
    if [[ -z "${found}" ]]; then
      echo "ERROR: unknown domain '${name}'" >&2
      exit 1
    fi
  done
else
  ENTRIES=("${DEFAULT_DOMAINS[@]}")
fi

unset CKPT WANDB_RUN_ID GPUS GPU DB_PATH_Y
mkdir -p logs

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run on the cluster login node." >&2
  exit 1
fi

echo "Submitting ${#ENTRIES[@]} Foundation Y job(s)"
echo "  CONFIG=${CONFIG}"
echo "  RUN_SUFFIX=${RUN_SUFFIX}"
echo ""

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r name db_path out_path <<<"${entry}"
  if [[ ! -d "${db_path}" ]]; then
    echo "SKIP ${name}: missing LMDB ${db_path}" >&2
    continue
  fi
  run_name="${name}_${RUN_SUFFIX}"
  export DB_PATH="${db_path}"
  export RUN_NAME="${run_name}"
  export CONFIG
  export OUT_PATH="${out_path}"
  export BATCH="${BATCH:-8}"
  export WORKERS="${WORKERS:-8}"

  job_id="$(
    sbatch --parsable \
      --chdir="${BRAVE_ROOT}" \
      --job-name="found-y-${name}" \
      --output="${BRAVE_ROOT}/logs/train-%j.log" \
      --error="${BRAVE_ROOT}/logs/train-%j.log" \
      "$@" \
      scripts/train.sbatch
  )" || {
    echo "ERROR: sbatch failed for ${name}" >&2
    exit 1
  }
  echo "  ${name}: job ${job_id}  RUN_NAME=${run_name}"
  echo "    log: ${BRAVE_ROOT}/logs/train-${job_id}.log"
done

echo ""
echo "Done. Watch: squeue -u \$USER"
echo "Foundation X: bash scripts/submit_foundation_x_train.sh"
