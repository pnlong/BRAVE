#!/usr/bin/env bash
# Stage an FSD50K whitelist into BRAVE-data/fsd50k_brave/<name>/audio_subset.
#
# Usage (from anywhere):
#   bash dataset_exploration/fsd50k/stage_named_subset.sh mechanical
#   bash dataset_exploration/fsd50k/stage_named_subset.sh synth_tonal
#   NAME=mechanical WHITELIST=path/to/custom.txt bash dataset_exploration/fsd50k/stage_named_subset.sh
#
# Env:
#   PARTITION   default: train,valid (comma-separated; maps to build_subset --partition)
#   METHOD      symlink|copy         default: symlink
#   WORKERS     staging workers      default: 8
#   OVERWRITE   1 to replace existing staged files
#   SUBMIT_PREPROCESS  1 to sbatch scripts/preprocess.sbatch after staging
#   BRAVE_STORAGE / BRAVE_ROOT

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
FSD_DIR="${BRAVE_ROOT}/dataset_exploration/fsd50k"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

NAME="${NAME:-${1:-}}"
if [[ -z "${NAME}" ]]; then
  echo "Usage: $0 <subset_name>" >&2
  echo "  known: mechanical | synth_tonal | water | …" >&2
  exit 1
fi

WHITELIST="${WHITELIST:-${FSD_DIR}/artifacts/whitelists/${NAME}.txt}"
if [[ ! -f "${WHITELIST}" ]]; then
  echo "ERROR: whitelist not found: ${WHITELIST}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-${BRAVE_STORAGE}/fsd50k_brave/${NAME}}"
AUDIO_SUBSET="${OUT_ROOT}/audio_subset"
PREPROCESSED="${OUT_ROOT}/preprocessed"
PARTITION_CSV="${PARTITION:-train,valid}"
METHOD="${METHOD:-symlink}"
WORKERS="${WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"
SUBMIT_PREPROCESS="${SUBMIT_PREPROCESS:-0}"

mkdir -p "${AUDIO_SUBSET}" "${OUT_ROOT}"

echo "=== stage FSD50K subset ==="
echo "  name:       ${NAME}"
echo "  whitelist:  ${WHITELIST}"
echo "  partitions: ${PARTITION_CSV}"
echo "  method:     ${METHOD}"
echo "  output:     ${AUDIO_SUBSET}"

cd "${FSD_DIR}"
IFS=',' read -r -a PARTS <<< "${PARTITION_CSV}"
for part in "${PARTS[@]}"; do
  part="$(echo "${part}" | xargs)"
  [[ -n "${part}" ]] || continue
  echo "--- partition ${part} ---"
  ARGS=(
    --whitelist "${WHITELIST}"
    --partition "${part}"
    --output-dir "${AUDIO_SUBSET}"
    --method "${METHOD}"
    --workers "${WORKERS}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
  fi
  python3 build_subset.py "${ARGS[@]}"
done

n_wav="$(find "${AUDIO_SUBSET}" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')"
echo ""
echo "=== staged ${n_wav} wav files under ${AUDIO_SUBSET} ==="
echo ""
echo "Stats (optional):"
echo "  cd ${FSD_DIR} && python3 subset_audio_stats.py --whitelist ${WHITELIST} --wav-root ${AUDIO_SUBSET}"
echo ""
echo "Preprocess:"
echo "  INPUT_PATH=${AUDIO_SUBSET} OUTPUT_PATH=${PREPROCESSED} \\"
echo "    sbatch ${BRAVE_ROOT}/scripts/preprocess.sbatch"

if [[ "${SUBMIT_PREPROCESS}" == "1" ]]; then
  mkdir -p "${BRAVE_ROOT}/logs"
  job_id="$(
    INPUT_PATH="${AUDIO_SUBSET}" OUTPUT_PATH="${PREPROCESSED}" \
      sbatch --parsable "${BRAVE_ROOT}/scripts/preprocess.sbatch"
  )"
  echo "Submitted preprocess job ${job_id}"
fi
