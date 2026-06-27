# Source from SLURM batch scripts (non-interactive shells skip conda in ~/.bashrc):
#   source "${BRAVE_ROOT}/scripts/slurm_env.sh"
#
# Compute nodes often set HOME=/tmp/home/$USER and do not mount AFS. Install a
# shared-storage env once on the login node:
#   ./scripts/setup_env_compute.sh
#
# Optional overrides:
#   BRAVE_PYTHON=/path/to/brave/env/bin/python
#   CONDA_BASE=/path/to/miniconda3
#   CONDA_ENV=brave

_slurm_env_name="${CONDA_ENV:-brave}"

if [[ -z "${BRAVE_ROOT:-}" ]]; then
  if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "${0}" ]]; then
    BRAVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  else
    echo "error: set BRAVE_ROOT before sourcing slurm_env.sh" >&2
    exit 1
  fi
  export BRAVE_ROOT
fi

_slurm_try_python() {
  local py="$1"
  [[ -n "${py}" && -x "${py}" ]] || return 1
  export BRAVE_PYTHON="${py}"
  export PATH="$(dirname "${py}"):${PATH}"
  return 0
}

_slurm_activate_brave() {
  local candidate
  local conda_base
  local -a python_candidates=(
    "${BRAVE_PYTHON:-}"
    "${BRAVE_ROOT}/../conda/envs/${_slurm_env_name}/bin/python"
    "/data/hai-res/p1long/conda/envs/${_slurm_env_name}/bin/python"
    "/data/scratch-fast/p1long/conda/envs/${_slurm_env_name}/bin/python"
    "${HOME}/.conda/envs/${_slurm_env_name}/bin/python"
    "/afs/csail.mit.edu/u/${USER}/.conda/envs/${_slurm_env_name}/bin/python"
    "${HOME}/miniconda3/envs/${_slurm_env_name}/bin/python"
    "/afs/csail.mit.edu/u/${USER}/miniconda3/envs/${_slurm_env_name}/bin/python"
  )

  for candidate in "${python_candidates[@]}"; do
    if _slurm_try_python "${candidate}"; then
      return 0
    fi
  done

  local -a conda_bases=(
    "${CONDA_BASE:-}"
    "${BRAVE_ROOT}/../miniconda3"
    "/data/hai-res/p1long/miniconda3"
    "${HOME}/miniconda3"
    "/afs/csail.mit.edu/u/${USER}/miniconda3"
  )

  for conda_base in "${conda_bases[@]}"; do
    [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]] || continue
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${_slurm_env_name}"
    if command -v python >/dev/null 2>&1; then
      export BRAVE_PYTHON="$(command -v python)"
      return 0
    fi
  done

  echo "error: could not find brave python env on shared storage" >&2
  echo "  HOME=${HOME:-<unset>} USER=${USER:-<unset>}" >&2
  echo "  BRAVE_ROOT=${BRAVE_ROOT}" >&2
  echo "  expected: ${BRAVE_ROOT}/../conda/envs/${_slurm_env_name}/bin/python" >&2
  echo "  Compute nodes cannot use AFS home; run once on the login node:" >&2
  echo "    ${BRAVE_ROOT}/scripts/setup_env_compute.sh" >&2
  return 1
}

_slurm_activate_brave || exit 1

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

echo "using python: ${BRAVE_PYTHON}"

unset _slurm_env_name _slurm_try_python _slurm_activate_brave
