#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-$ROOT/third_party/LIBERO}"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="$ROOT/.cache/libero-research"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
export XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$LIBERO_CONFIG_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$LIBERO_CONFIG_ROOT/numba"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
SELECTED_GPU="${GPU:-2}"
[[ "$SELECTED_GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be a nvidia-smi numeric index' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SELECTED_GPU" --query-gpu=uuid --format=csv,noheader)"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *$'\n'* ]] || exit 2
export MUJOCO_EGL_DEVICE_ID="$SELECTED_GPU"
export RUN="${RUN:-$ROOT/experiment/h2_states_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "Supplemental H2 state collection: $RUN; selected GPU UUID: $CUDA_VISIBLE_DEVICES"
LIBERO_PYTHON="$PYTHON" bash benchmarks/libero/run_smoke.sh env
SPLIT="${E1_RUN:-$ROOT/experiment/h1h2_20261112_223445_299984}/manifests/split.json"
model=(--ckpt-dir "${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}" --ckpt-name "${CKPT_NAME:-checkpoint_step_10690.safetensors}")
for subset in discovery calibration; do
  "$PYTHON" -m benchmarks.libero.research collect "${model[@]}" --split "$SPLIT" \
    --selected-split "$subset" --seeds 43 --episodes 10 --output "$RUN/state_bank/$subset"
  "$PYTHON" -m benchmarks.libero.research t0 "${model[@]}" --split "$SPLIT" \
    --bank "$RUN/state_bank/$subset" --seeds 43 44 45 --output "$RUN/diagnostics/t0_$subset"
done
"$PYTHON" - <<'PY'
import os
from collections import defaultdict
from pathlib import Path

from openwam.research.protocol import read_sealed, write_json

report = {}
for subset in ('discovery', 'calibration'):
    bank = read_sealed(Path(os.environ['RUN']) / 'state_bank' / subset / 'manifest.json')
    groups = defaultdict(set)
    for state in bank['states']:
        phase = 0 if state['progress'] < 0.35 else (1 if state['progress'] < 0.65 else 2)
        groups[f"{state['suite']}:{state['task_id']}"].add(phase)
    report[subset] = {task: sorted(phases) for task, phases in sorted(groups.items())}
write_json(Path(os.environ['RUN']) / 'coverage.json', report)
print('Coverage:', report)
print('Supplemental collection complete. Coverage is not guaranteed; inspect before freezing E2 selection. No E2 perturbations have run.')
PY
