#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="$ROOT/.cache/libero-research"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT" XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$LIBERO_CONFIG_ROOT/matplotlib" NUMBA_CACHE_DIR="$LIBERO_CONFIG_ROOT/numba"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
RUN="${RUN:-$ROOT/experiment/e2_robustness_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "H2 robustness PID=$$ run=$RUN"
if [[ "${PREPARE_ONLY:-0}" != 1 ]]; then
  SELECTED_GPU="${GPU:-4}"
  [[ "$SELECTED_GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical nvidia-smi index'; exit 2; }
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SELECTED_GPU" --query-gpu=uuid --format=csv,noheader)"
  [[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *$'\n'* ]] || exit 2
  export MUJOCO_EGL_DEVICE_ID="$SELECTED_GPU"
  echo "Physical GPU $SELECTED_GPU -> $CUDA_VISIBLE_DEVICES"
fi
arguments=(--run "$RUN" --hours "${HOURS:-10}")
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then arguments+=(--prepare-only); fi
"$PYTHON" scripts/research_e2_robustness.py "${arguments[@]}" "$@"
