#!/usr/bin/env bash
# Fresh single-GPU replication. Reference runs are inputs, never reused results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RESEARCH_PYTHON="${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}"
export RUN="${RUN:-$ROOT/experiment/e2_reproduction_$(date +%Y%m%d_%H%M%S)_$$}"
export LIBERO_PATH="$ROOT/third_party/LIBERO"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="$ROOT/.cache/libero-research"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT" XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$LIBERO_CONFIG_ROOT/matplotlib" NUMBA_CACHE_DIR="$LIBERO_CONFIG_ROOT/numba"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
exec "$RESEARCH_PYTHON" scripts/reproduce_e2_robustness.py --run "$RUN" "$@"
