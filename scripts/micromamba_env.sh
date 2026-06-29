# Resolve micromamba on hai-res (binary is often ~/.local/bin/micromamba, not
# $MAMBA_ROOT_PREFIX/bin/micromamba). Source before micromamba activate:
#   source "${BRAVE_ROOT}/scripts/micromamba_env.sh"
#
# Optional overrides:
#   MAMBA_EXE=/path/to/micromamba
#   MAMBA_ROOT_PREFIX=/path/to/micromamba/root

set -euo pipefail

# hai-res nodes often cannot write/read AFS home; keep mamba config on shared storage.
_hai_res_home="/data/hai-res/${USER}"
if [[ -d "${_hai_res_home}" ]]; then
  export XDG_CONFIG_HOME="${_hai_res_home}/.config"
  export XDG_CACHE_HOME="${_hai_res_home}/.cache"
  mkdir -p "${XDG_CONFIG_HOME}/mamba" "${XDG_CACHE_HOME}"
fi
unset _hai_res_home

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/hai-res/${USER}/micromamba}"
export MAMBA_ROOT_PREFIX

_brave_find_micromamba() {
  local candidate
  local -a candidates=(
    "${MAMBA_EXE:-}"
    "${MAMBA_ROOT_PREFIX}/bin/micromamba"
    "/data/hai-res/${USER}/.local/bin/micromamba"
    "${HOME}/.local/bin/micromamba"
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    echo "${candidate}"
    return 0
  done
  local path_bin
  path_bin="$(type -P micromamba 2>/dev/null || true)"
  if [[ -n "${path_bin}" && -x "${path_bin}" ]]; then
    echo "${path_bin}"
    return 0
  fi
  return 1
}

if ! MAMBA_EXE="$(_brave_find_micromamba)"; then
  echo "error: micromamba not found" >&2
  echo "  expected one of:" >&2
  echo "    /data/hai-res/${USER}/.local/bin/micromamba" >&2
  echo "    \${MAMBA_ROOT_PREFIX}/bin/micromamba" >&2
  echo "  or set MAMBA_EXE to your micromamba binary" >&2
  return 1 2>/dev/null || exit 1
fi

export MAMBA_EXE

if declare -f micromamba >/dev/null 2>&1; then
  : # shell hook already loaded (e.g. from ~/.bashrc)
else
  # shellcheck disable=SC1090
  __mamba_setup="$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}" 2>/dev/null || true)"
  if [[ -n "${__mamba_setup}" ]]; then
    eval "${__mamba_setup}"
  else
    eval "$("${MAMBA_EXE}" shell hook -s bash -p "${MAMBA_ROOT_PREFIX}")"
  fi
  unset __mamba_setup
fi

unset -f _brave_find_micromamba
