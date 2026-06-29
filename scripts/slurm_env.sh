# Source from SLURM batch scripts (non-interactive shells skip ~/.bashrc):
#   source "${BRAVE_ROOT}/scripts/slurm_env.sh"
#
# Expects micromamba on hai-res shared storage (visible on compute nodes).
#
# Optional overrides:
#   MAMBA_EXE=/path/to/micromamba
#   MAMBA_ROOT_PREFIX=/path/to/micromamba/root
#   MAMBA_ENV=brave

set -euo pipefail

: "${BRAVE_ROOT:?set BRAVE_ROOT before sourcing slurm_env.sh}"

MAMBA_ENV="${MAMBA_ENV:-brave}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/micromamba_env.sh"
micromamba activate "${MAMBA_ENV}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

echo "micromamba: ${MAMBA_ROOT_PREFIX} (${MAMBA_ENV})"
echo "python: $(command -v python)"

# Build --gpu args for RAVE/scripts/train.py (DEFINE_multi_integer gpu).
# Set GPUS=0,1 or sbatch --gres=gpu:N and leave GPUS unset to use 0..N-1.
# Legacy single-GPU: GPU=0
slurm_build_gpu_train_args() {
  local -n _out=$1
  _out=()
  if [[ -n "${GPUS:-}" ]]; then
    local _ids _g
    IFS=',' read -ra _ids <<< "${GPUS}"
    for _g in "${_ids[@]}"; do
      _out+=(--gpu="${_g}")
    done
  elif [[ -n "${GPU:-}" ]]; then
    _out+=(--gpu="${GPU}")
  else
    local _n="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
    local _i
    for ((_i = 0; _i < _n; _i++)); do
      _out+=(--gpu="${_i}")
    done
  fi
}
