#!/usr/bin/env bash
# Stage JSUT WAVs into $BRAVE_STORAGE/jsut/audio_subset (flat symlinks).
#
# Usage:
#   bash dataset_exploration/jsut/stage_subset.sh
#   SUBSETS=basic5000,travel1000 bash dataset_exploration/jsut/stage_subset.sh
#   OVERWRITE=1 SUBMIT_PREPROCESS=1 bash dataset_exploration/jsut/stage_subset.sh
#
# Env:
#   JSUT_ROOT          default /data/hai-res/shared/p1long/JSUT/jsut_ver1.1
#   SUBSETS            comma-separated subset names, or "all" (default)
#   METHOD             symlink|copy  (default symlink)
#   OVERWRITE          1 to replace existing staged names
#   SUBMIT_PREPROCESS  1 to sbatch scripts/preprocess.sbatch

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

JSUT_ROOT="${JSUT_ROOT:-/data/hai-res/shared/p1long/JSUT/jsut_ver1.1}"
OUT_ROOT="${BRAVE_STORAGE}/jsut"
AUDIO_SUBSET="${OUT_ROOT}/audio_subset"
PREPROCESSED="${OUT_ROOT}/preprocessed"
METHOD="${METHOD:-symlink}"
OVERWRITE="${OVERWRITE:-0}"
SUBMIT_PREPROCESS="${SUBMIT_PREPROCESS:-0}"
SUBSETS_CSV="${SUBSETS:-all}"

if [[ ! -d "${JSUT_ROOT}" ]]; then
  echo "ERROR: JSUT root not found: ${JSUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${AUDIO_SUBSET}" "${OUT_ROOT}"

if [[ "${SUBSETS_CSV}" == "all" ]]; then
  mapfile -t SUBSET_LIST < <(
    find "${JSUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  )
else
  IFS=',' read -r -a SUBSET_LIST <<< "${SUBSETS_CSV}"
fi

echo "=== stage JSUT ==="
echo "  JSUT_ROOT:  ${JSUT_ROOT}"
echo "  subsets:    ${SUBSET_LIST[*]}"
echo "  method:     ${METHOD}"
echo "  output:     ${AUDIO_SUBSET}"

staged=0
skipped=0
missing=0

for subset in "${SUBSET_LIST[@]}"; do
  subset="$(echo "${subset}" | xargs)"
  [[ -n "${subset}" ]] || continue
  wav_dir="${JSUT_ROOT}/${subset}/wav"
  if [[ ! -d "${wav_dir}" ]]; then
    echo "WARNING: missing ${wav_dir}" >&2
    missing=$((missing + 1))
    continue
  fi
  shopt -s nullglob
  for src in "${wav_dir}"/*.wav; do
    base="$(basename "${src}")"
    # Prefix with subset to avoid collisions across folders.
    dst="${AUDIO_SUBSET}/${subset}__${base}"
    if [[ -e "${dst}" || -L "${dst}" ]]; then
      if [[ "${OVERWRITE}" == "1" ]]; then
        rm -f "${dst}"
      else
        skipped=$((skipped + 1))
        continue
      fi
    fi
    if [[ "${METHOD}" == "copy" ]]; then
      cp -f "${src}" "${dst}"
    else
      ln -s "$(readlink -f "${src}")" "${dst}"
    fi
    staged=$((staged + 1))
  done
  shopt -u nullglob
done

n_wav="$(find "${AUDIO_SUBSET}" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')"
echo ""
echo "=== staged=${staged} skipped_existing=${skipped} missing_subsets=${missing} total_wav=${n_wav} ==="
echo "  audio_subset: ${AUDIO_SUBSET}"
echo ""
echo "Preprocess (48 kHz → 44.1 kHz mono via RAVE preprocess):"
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
