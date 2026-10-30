#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/outputs/robotwin/environment.sh" ]]; then
    source "${REPO_ROOT}/outputs/robotwin/environment.sh"
fi
exec "${SERVER_PYTHON:-${REPO_ROOT}/.venv-openwam/bin/python}" \
    -m benchmarks.robotwin.compare_joint "$@"
