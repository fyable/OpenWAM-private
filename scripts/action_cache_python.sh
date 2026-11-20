#!/usr/bin/env bash
# Transparent interpreter adapter for the pinned LIBERO evaluator.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTION_CACHE_PYTHON:-$ROOT/.venv-openwam/bin/python}"
if [[ "${1:-}" == "$ROOT/scripts/deploy.py" ]]; then
    shift
    cd "$ROOT"
    exec "$PYTHON" -m benchmarks.action_cache.server "$@"
fi
exec "$PYTHON" "$@"
