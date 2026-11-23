#!/usr/bin/env bash
# New independent experiment: use all selected GPUs until one absolute deadline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LIBERO_PATH="$ROOT/third_party/LIBERO"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="$ROOT/.cache/libero-research"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT" XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$LIBERO_CONFIG_ROOT/matplotlib" NUMBA_CACHE_DIR="$LIBERO_CONFIG_ROOT/numba"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export E2_CPU_THREADS_PER_WORKER="${E2_CPU_THREADS_PER_WORKER:-4}"
export OMP_NUM_THREADS="$E2_CPU_THREADS_PER_WORKER" MKL_NUM_THREADS="$E2_CPU_THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$E2_CPU_THREADS_PER_WORKER"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export E2_MEMORY_MODE="${E2_MEMORY_MODE:-text-offload}"
exec "${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}" scripts/reproduce_e2_fresh.py "$@"
