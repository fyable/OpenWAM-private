#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-e1}"
case "$MODE" in
  check|e1|calibration|e2) ;;
  *) echo 'Usage: bash scripts/research_alpha_libero.sh [check|e1|calibration|e2]' >&2; exit 2 ;;
esac
if [[ "$MODE" == calibration ]]; then
  : "${E1_RUN:?Set E1_RUN to the completed E1 directory}"
  [[ -f "$E1_RUN/manifests/split.json" ]] || { echo 'Missing E1 split' >&2; exit 2; }
fi
if [[ "$MODE" == e2 ]]; then
  : "${E1_RUN:?Set E1_RUN to the completed E1 directory}"
  : "${SELECTION:?Set SELECTION to a sealed Discovery selection JSON}"
  : "${CALIBRATION_BANK:?Set CALIBRATION_BANK to the existing Calibration bank}"
  : "${CALIBRATION_T0:?Set CALIBRATION_T0 to its passing T0 certificate}"
fi
PYTHON="${RESEARCH_PYTHON:-$ROOT/.venv-openwam/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-$ROOT/third_party/LIBERO}"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="${LIBERO_CONFIG_ROOT:-$ROOT/.cache/libero-research}"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LIBERO_CONFIG_ROOT/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$LIBERO_CONFIG_ROOT/numba}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
  echo 'GPU must be a single numeric GPU index' >&2
  exit 2
fi
export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
CKPT_NAME="${CKPT_NAME:-checkpoint_step_10690.safetensors}"
export RUN="${RUN:-$ROOT/experiment/h1h2_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "Research output: $RUN"
LIBERO_PYTHON="$PYTHON" bash benchmarks/libero/run_smoke.sh env
"$PYTHON" -c 'import torch; from openwam.deploy.engine import JointInferenceEngine; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name())'
if [[ "$MODE" == check ]]; then exit 0; fi
research() { "$PYTHON" -m benchmarks.libero.research "$@"; }
model=(--ckpt-dir "$CKPT_DIR" --ckpt-name "$CKPT_NAME")
if [[ "$MODE" == e1 ]]; then
  "$PYTHON" - <<'PY'
import os
from libero.libero import benchmark
from openwam.research.protocol import SUITES, write_json

tasks = {}
for name in SUITES:
    suite = benchmark.get_benchmark_dict()[name]()
    tasks[name] = [suite.get_task(index).name for index in range(10)]
write_json(os.path.join(os.environ['RUN'], 'manifests/tasks.json'), tasks)
PY
  research split --task-map "$RUN/manifests/tasks.json" --output "$RUN/manifests/split.json"
  common=("${model[@]}" --split "$RUN/manifests/split.json")
  research collect "${common[@]}" --selected-split discovery --seeds 42 --episodes 10 --output "$RUN/state_bank/discovery"
  bank=(--bank "$RUN/state_bank/discovery")
  research t0 "${common[@]}" "${bank[@]}" --seeds 42 43 44 --output "$RUN/diagnostics/t0_discovery"
  LAYERS="$("$PYTHON" -c 'import json,os; print(json.load(open(os.environ["RUN"]+"/state_bank/discovery/model.json"))["num_layers"])')"
  research plan --experiment e1 --layers "$LAYERS" --split "$RUN/manifests/split.json" --output "$RUN/manifests/e1.json"
  research smoke "${common[@]}" "${bank[@]}" --plan "$RUN/manifests/e1.json" --method-id E1_L0_S0_action --seeds 42 --output "$RUN/diagnostics/e1_smoke"
  research replay "${common[@]}" "${bank[@]}" --plan "$RUN/manifests/e1.json" --t0 "$RUN/diagnostics/t0_discovery/t0_certificate.json" --seeds 42 --output "$RUN/diagnostics/e1_screen"
  research summarize --runs "$RUN/diagnostics/e1_screen" --output "$RUN/tables/e1"
  echo "E1 complete. Freeze an E1-informed selection before E2; see benchmarks/libero/RESEARCH.md section 4."
elif [[ "$MODE" == calibration ]]; then
  mkdir -p "$RUN/manifests"
  cp -n "$E1_RUN/manifests/split.json" "$RUN/manifests/split.json"
  common=("${model[@]}" --split "$RUN/manifests/split.json")
  research collect "${common[@]}" --selected-split calibration --seeds 42 --episodes 10 --output "$RUN/state_bank/calibration"
  bank=(--bank "$RUN/state_bank/calibration")
  research t0 "${common[@]}" "${bank[@]}" --seeds 42 43 44 --output "$RUN/diagnostics/t0_calibration"
  LAYERS="$("$PYTHON" -c 'import json,os; print(json.load(open(os.environ["RUN"]+"/state_bank/calibration/model.json"))["num_layers"])')"
  research plan --experiment e1 --layers "$LAYERS" --selected-split calibration --split "$RUN/manifests/split.json" --output "$RUN/manifests/e1_calibration.json"
  research smoke "${common[@]}" "${bank[@]}" --plan "$RUN/manifests/e1_calibration.json" --method-id E1_L0_S0_action --seeds 42 --output "$RUN/diagnostics/e1_calibration_smoke"
  research replay "${common[@]}" "${bank[@]}" --plan "$RUN/manifests/e1_calibration.json" --t0 "$RUN/diagnostics/t0_calibration/t0_certificate.json" --seeds 42 --output "$RUN/diagnostics/e1_calibration"
  research summarize --runs "$RUN/diagnostics/e1_calibration" --output "$RUN/tables/e1_calibration"
  echo 'Calibration E1 complete; held-out tasks were not used. E2 still requires a frozen Discovery selection.'
else
  common=("${model[@]}" --split "$E1_RUN/manifests/split.json")
  research features "${common[@]}" --bank "$CALIBRATION_BANK" --selected-split calibration --selection "$SELECTION" --t0 "$CALIBRATION_T0" --seeds 42 --output "$RUN/diagnostics/calibration_features"
  research features "${common[@]}" --bank "$E1_RUN/state_bank/discovery" --selected-split discovery --selection "$SELECTION" --t0 "$E1_RUN/diagnostics/t0_discovery/t0_certificate.json" --seeds 42 --output "$RUN/diagnostics/discovery_features"
  LAYERS="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_layers"])' "$E1_RUN/state_bank/discovery/model.json")"
  research plan --experiment e2 --layers "$LAYERS" --split "$E1_RUN/manifests/split.json" --selection "$SELECTION" --calibration "$RUN/diagnostics/calibration_features/calibration.json" --output "$RUN/manifests/e2.json"
  research replay "${common[@]}" --bank "$E1_RUN/state_bank/discovery" --plan "$RUN/manifests/e2.json" --t0 "$E1_RUN/diagnostics/t0_discovery/t0_certificate.json" --seeds 42 --output "$RUN/diagnostics/e2_screen"
  research summarize --runs "$RUN/diagnostics/e2_screen" --features "$RUN/diagnostics/discovery_features/features.json" --calibration "$RUN/diagnostics/calibration_features/calibration.json" --output "$RUN/tables/e2"
fi
