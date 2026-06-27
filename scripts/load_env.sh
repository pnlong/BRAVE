# Source optional secrets from BRAVE_ROOT/.env (gitignored).
# Usage (after BRAVE_ROOT is set):
#   source "${BRAVE_ROOT}/scripts/load_env.sh"

: "${BRAVE_ROOT:?set BRAVE_ROOT before sourcing load_env.sh}"

_ENV_FILE="${BRAVE_ROOT}/.env"
if [[ -f "${_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_ENV_FILE}"
  set +a
fi
unset _ENV_FILE
