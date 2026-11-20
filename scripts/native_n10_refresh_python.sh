#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-openwam/bin/python"
if [[ "${1:-}" == "$ROOT/scripts/deploy.py" ]]; then
    shift
    cd "$ROOT"
    exec "$PYTHON" -m scripts.native_n10_refresh_server "$@"
fi
exec "$PYTHON" "$@"
