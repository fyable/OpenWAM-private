#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec "${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}" -m benchmarks.libero.reproduce_paraflow_110 "${1:-prepare}"
