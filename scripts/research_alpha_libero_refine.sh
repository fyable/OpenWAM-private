#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export E1_RUN="${E1_RUN:-$ROOT/experiment/h1h2_20261112_223445_299984}"
export REFINE_MODE="${REFINE_MODE:-discovery}"
case "$REFINE_MODE" in
  discovery) BANK_RUN="$E1_RUN"; BANK_SPLIT=discovery ;;
  representatives) BANK_RUN="${CALIBRATION_RUN:-$ROOT/experiment/h1h2_20261113_005818_705220}"; BANK_SPLIT=calibration ;;
  *) echo 'REFINE_MODE must be discovery or representatives' >&2; exit 2 ;;
esac
export RUN="${RUN:-$ROOT/experiment/e1_refine_$(date +%Y%m%d_%H%M%S)_$$}"
PYTHON="${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
SELECTED_GPU="${GPU:-6}"
if [[ "$SELECTED_GPU" =~ ^[0-9]+$ ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SELECTED_GPU" --query-gpu=uuid --format=csv,noheader)"
  [[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *$'\n'* ]] || exit 2
else
  CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
fi
export CUDA_VISIBLE_DEVICES
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/libero-research/matplotlib}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "Refinement output: $RUN; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PYTHON" - <<'PY'
import copy
import os
from pathlib import Path

from openwam.research.protocol import read_sealed, seal, write_json

previous = Path(os.environ['E1_RUN'])
plan = read_sealed(previous / 'manifests/e1.json')
plan.pop('sha256')
assert plan['selected_split'] == 'discovery'
methods = []
for original in plan['methods']:
    if original['method_id'] not in {'E1_L0_S2_action', 'E1_L3_S2_action'}:
        continue
    for layer in original['intervention']['layer_ids']:
        method = copy.deepcopy(original)
        method['method_id'] = f'refine_layer{layer}_late_action'
        method['intervention']['layer_ids'] = [layer]
        methods.append(method)
if os.environ['REFINE_MODE'] == 'representatives':
    methods = []
    for layer in (0, 29):
        for step in (1, 5, 9):
            methods.append({'method_id': f'representative_layer{layer}_step{step}',
                            'intervention': {'layer_ids': [layer], 'step_ids': [step],
                                             'kind': 'gate', 'gate': 0.0,
                                             'query_modality': 'action', 'source_modality': 'future_video'}})
    plan['selected_split'] = 'calibration'
    plan['episode_ids'] = list(range(10))
    plan['selection_rule'] = 'Discovery refinement: maximum episode-mean late RMSE within layers 0-7 and 23-29; layers 0 and 29'
assert 0 < len(methods) <= 20
plan['methods'] = methods
plan['fitted_split'] = 'discovery'
plan['purpose'] = 'Single-layer late-step refinement of Discovery-selected shallow/deep groups'
if os.environ['REFINE_MODE'] == 'representatives':
    plan['purpose'] = 'Calibration validation of Discovery-selected layers 0 and 29 at steps 1, 5, 9'
write_json(Path(os.environ['RUN']) / 'refine_plan.json', seal(plan))
print(f'Frozen {len(methods)} single-layer configurations; no Calibration or held-out ranking used')
PY
"$PYTHON" -m benchmarks.libero.research replay \
  --ckpt-dir "${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}" \
  --ckpt-name "${CKPT_NAME:-checkpoint_step_10690.safetensors}" \
  --split "$BANK_RUN/manifests/split.json" \
  --bank "$BANK_RUN/state_bank/$BANK_SPLIT" \
  --plan "$RUN/refine_plan.json" \
  --t0 "$BANK_RUN/diagnostics/t0_$BANK_SPLIT/t0_certificate.json" \
  --seeds 42 --output "$RUN/diagnostics/refine"
"$PYTHON" -m benchmarks.libero.research summarize \
  --runs "$RUN/diagnostics/refine" --output "$RUN/tables/refine"
echo "Refinement mode $REFINE_MODE complete. This is gate validation, not E2 value perturbation."
