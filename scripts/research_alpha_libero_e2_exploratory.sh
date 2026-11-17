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
export BANK_RUN="${BANK_RUN:-$ROOT/experiment/h2_states_20261113_175408_491989}"
export E1_RUN="${E1_RUN:-$ROOT/experiment/h1h2_20261112_223445_299984}"
export REFINE_RUN="${REFINE_RUN:-$ROOT/experiment/e1_refine_20261113_152808_630612}"
export RUN="${RUN:-$ROOT/experiment/e2_exploratory_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "E2 exploratory output: $RUN"
"$PYTHON" - <<'PY'
import os
from pathlib import Path

from benchmarks.libero.research import source_identity
from openwam.research.exploratory import PROTOCOL, select_states, source_changes, validate_coverage
from openwam.research.protocol import file_hash, read_sealed, seal, write_json

root = Path(os.environ['RUN'])
split = read_sealed(Path(os.environ['E1_RUN']) / 'manifests/split.json')
current = source_identity()
selection = {'fitted_split': 'discovery', 'task_split_sha256': split['sha256'],
             'sampling_protocol': PROTOCOL, 'layer_ids': [0, 29], 'step_ids': [1, 5, 9],
             'state_ids_by_split': {}, 'sampling_coverage': {}, 'bank_sha256_by_split': {},
             'layouts': {}, 'representative_evidence': {}}
for subset in ('discovery', 'calibration'):
    directory = Path(os.environ['BANK_RUN']) / 'state_bank' / subset
    bank = read_sealed(directory / 'manifest.json')
    old = read_sealed(directory / 'model.json')
    if bank['task_split_sha256'] != split['sha256'] or bank['model_sha256'] != old['sha256']:
        raise ValueError('Bank provenance mismatch')
    if any(state['split'] != subset for state in bank['states']):
        raise ValueError('Mixed split bank')
    ids, coverage = select_states(bank['states'])
    validate_coverage([state for state in bank['states'] if state['state_id'] in ids], PROTOCOL)
    selection['state_ids_by_split'][subset] = ids
    selection['sampling_coverage'][subset] = coverage
    selection['bank_sha256_by_split'][subset] = bank['sha256']
    write_json(root / 'manifests' / f'source_review_{subset}.json', seal({
        'old_model_sha256': old['sha256'], 'bank_sha256': bank['sha256'],
        'new_source_sha256': current['source_sha256'], 'changes': source_changes(old, current),
        'reason': 'Explicit exploratory sampling; unchanged model operators/assets/config; fresh T0 required'}))
    print(subset, 'selected states:', len(ids), 'missing phases:',
          [(row['suite'], row['task_id'], row['missing_phases']) for row in coverage if row['missing_phases']])
for path in sorted((Path(os.environ['REFINE_RUN']) / 'diagnostics/refine').glob('*.json')):
    import json
    row = json.loads(path.read_text())
    if row.get('status') != 'completed' or row.get('split') != 'discovery':
        continue
    for layer in ('0', '29'):
        if layer in row.get('layouts', {}) and layer not in selection['layouts']:
            selection['layouts'][layer] = row['layouts'][layer]
            selection['representative_evidence'][layer] = {'file': str(path), 'sha256': file_hash(path)}
if set(selection['layouts']) != {'0', '29'}:
    raise ValueError('Missing Discovery representative layouts')
write_json(root / 'manifests/selection.json', seal(selection))
print('Selection frozen before E2 outcomes; original banks unchanged.')
PY
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
SELECTED_GPU="${GPU:-4}"
[[ "$SELECTED_GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be a nvidia-smi index'; exit 2; }
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SELECTED_GPU" --query-gpu=uuid --format=csv,noheader)"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *$'\n'* ]] || exit 2
export MUJOCO_EGL_DEVICE_ID="$SELECTED_GPU"
echo "Physical GPU $SELECTED_GPU -> $CUDA_VISIBLE_DEVICES"
research() { "$PYTHON" -m benchmarks.libero.research "$@"; }
common=(--ckpt-dir "${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}" --ckpt-name "${CKPT_NAME:-checkpoint_step_10690.safetensors}" --split "$E1_RUN/manifests/split.json")
for subset in calibration discovery; do
  bank=(--bank "$BANK_RUN/state_bank/$subset" --source-review "$RUN/manifests/source_review_$subset.json")
  research t0 "${common[@]}" "${bank[@]}" --seeds 43 44 45 --output "$RUN/diagnostics/t0_$subset"
  research features "${common[@]}" "${bank[@]}" --selected-split "$subset" --selection "$RUN/manifests/selection.json" --t0 "$RUN/diagnostics/t0_$subset/t0_certificate.json" --seeds 43 --output "$RUN/diagnostics/features_$subset"
done
for subset in discovery calibration; do
  bank=(--bank "$BANK_RUN/state_bank/$subset" --source-review "$RUN/manifests/source_review_$subset.json")
  research plan --experiment e2 --layers 30 --selected-split "$subset" --split "$E1_RUN/manifests/split.json" --selection "$RUN/manifests/selection.json" --calibration "$RUN/diagnostics/features_calibration/calibration.json" --output "$RUN/manifests/e2_$subset.json"
  research smoke "${common[@]}" "${bank[@]}" --plan "$RUN/manifests/e2_$subset.json" --method-id E2_L0_S1_G0_sign-1 --seeds 43 --output "$RUN/diagnostics/smoke_$subset"
  research replay "${common[@]}" "${bank[@]}" --plan "$RUN/manifests/e2_$subset.json" --t0 "$RUN/diagnostics/t0_$subset/t0_certificate.json" --seeds 43 --output "$RUN/diagnostics/e2_$subset"
  research summarize --runs "$RUN/diagnostics/e2_$subset" --features "$RUN/diagnostics/features_$subset/features.json" --calibration "$RUN/diagnostics/features_calibration/calibration.json" --output "$RUN/tables/e2_$subset"
done
echo 'E2 exploratory complete. Progress proxies are not contact labels; no held-out, closed-loop or speedup claim.'
