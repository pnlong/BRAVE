#!/usr/bin/env bash
# Copy the login-node `brave` conda env to hai-res shared storage for SLURM jobs.
# Compute nodes use HOME=/tmp/... and often cannot see AFS (/afs/csail.mit.edu/...).
#
# Usage (from login node, once):
#   ./scripts/setup_env_compute.sh
#   ./scripts/setup_env_compute.sh --force   # re-clone

set -euo pipefail

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

BRAVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPUTE_ENV="${COMPUTE_ENV:-/data/hai-res/p1long/conda/envs/brave}"
SOURCE_ENV="${SOURCE_ENV:-brave}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--force]"
      echo "  COMPUTE_ENV=${COMPUTE_ENV}"
      echo "  SOURCE_ENV=${SOURCE_ENV}"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -x "${COMPUTE_ENV}/bin/python" && "${FORCE}" -eq 0 ]]; then
  echo "compute env already exists: ${COMPUTE_ENV}"
  "${COMPUTE_ENV}/bin/python" -c "import torch, torchaudio, rave; print('ok', torch.__version__)"
  exit 0
fi

if [[ "${FORCE}" -eq 1 && -d "${COMPUTE_ENV}" ]]; then
  echo "removing existing ${COMPUTE_ENV}"
  rm -rf "${COMPUTE_ENV}"
fi

mkdir -p "$(dirname "${COMPUTE_ENV}")"
echo "cloning ${SOURCE_ENV} -> ${COMPUTE_ENV} (this may take several minutes)"
conda create --clone "${SOURCE_ENV}" -p "${COMPUTE_ENV}" -y

echo "verifying compute env"
PYTHONPATH="${BRAVE_ROOT}/RAVE${PYTHONPATH:+:${PYTHONPATH}}" \
  "${COMPUTE_ENV}/bin/python" -c "import torch, torchaudio, rave; print('ok', torch.__version__)"

echo "done: ${COMPUTE_ENV}"
echo "SLURM jobs will use this env automatically via scripts/slurm_env.sh"
