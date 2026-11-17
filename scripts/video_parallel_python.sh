#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${VIDEO_PARALLEL_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
if [[ "${1:-}" == "${ROOT}/scripts/deploy.py" ]]; then
    shift
    cd "${ROOT}"
    exec "${PYTHON}" -m benchmarks.video_parallel.server "$@"
fi
exec "${PYTHON}" "$@"
