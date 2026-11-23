#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LIBERO_PATH="$ROOT/third_party/LIBERO"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="${LIBERO_CONFIG_ROOT:-$ROOT/.cache/action-cache-libero}"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT" XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$LIBERO_CONFIG_ROOT/matplotlib" NUMBA_CACHE_DIR="$LIBERO_CONFIG_ROOT/numba"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "${ACTION_CACHE_PYTHON:-$ROOT/.venv-openwam/bin/python}" -m benchmarks.action_cache.experiment "$@"
