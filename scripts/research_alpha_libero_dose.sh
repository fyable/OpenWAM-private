#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CALIBRATION_RUN="${CALIBRATION_RUN:-$ROOT/experiment/h1h2_20261113_005818_705220}"
export RUN="${RUN:-$ROOT/experiment/e1_dose_$(date +%Y%m%d_%H%M%S)_$$}"
PYTHON="${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
SELECTED_GPU="${GPU:-0}"
if [[ "$SELECTED_GPU" =~ ^[0-9]+$ ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SELECTED_GPU" --query-gpu=uuid --format=csv,noheader)"
  [[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *$'\n'* ]] || {
    echo 'Unable to resolve requested nvidia-smi GPU index to one UUID' >&2
    exit 2
  }
else
  CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
fi
export CUDA_VISIBLE_DEVICES
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/libero-research/matplotlib}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "Dose validation output: $RUN"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; inference uses logical cuda:0"
"$PYTHON" - <<'PY'
import os
import torch

print(f"GPU selection: GPU={os.environ.get('GPU', '(unset)')}; gpu={os.environ.get('gpu', '(unset)')}")
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
free, total = torch.cuda.mem_get_info(0)
print(f"Logical cuda:0: {torch.cuda.get_device_properties(0)}; free={free / 2**30:.2f} GiB; total={total / 2**30:.2f} GiB")
PY
"$PYTHON" - <<'PY'
import copy
import os
from pathlib import Path

from openwam.research.protocol import read_sealed, seal, write_json

previous = Path(os.environ['CALIBRATION_RUN'])
plan = read_sealed(previous / 'manifests/e1_calibration.json')
plan.pop('sha256')
selected = {'E1_L0_S0_action', 'E1_L3_S0_action', 'E1_L0_S2_action', 'E1_L3_S2_action'}
methods = []
for method in plan['methods']:
    if method['method_id'] not in selected:
        continue
    for gate in (0.0, 0.5, 0.75, 1.0):
        candidate = copy.deepcopy(method)
        candidate['method_id'] += f'_gate{gate}'
        candidate['intervention']['gate'] = gate
        methods.append(candidate)
assert len(methods) == 16
plan['methods'] = methods
plan['purpose'] = 'Calibration exploratory dose validation of Discovery-selected early/late shallow/deep groups'
write_json(Path(os.environ['RUN']) / 'dose_plan.json', seal(plan))
PY
"$PYTHON" -m benchmarks.libero.research replay \
  --ckpt-dir "${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}" \
  --ckpt-name "${CKPT_NAME:-checkpoint_step_10690.safetensors}" \
  --split "$CALIBRATION_RUN/manifests/split.json" \
  --bank "$CALIBRATION_RUN/state_bank/calibration" \
  --plan "$RUN/dose_plan.json" \
  --t0 "$CALIBRATION_RUN/diagnostics/t0_calibration/t0_certificate.json" \
  --seeds 42 --output "$RUN/diagnostics/dose"
"$PYTHON" -m benchmarks.libero.research summarize \
  --runs "$RUN/diagnostics/dose" --output "$RUN/tables/dose"
echo 'Dose validation complete. No closed-loop or speedup claim; inspect dose curves before E2.'
