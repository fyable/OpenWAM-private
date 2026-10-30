#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec "${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}" -m benchmarks.libero.compare_original_paraflow "${1:-plan}"
