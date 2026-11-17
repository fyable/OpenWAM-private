#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export GPUS="${GPU:-0}"
export REPLICAS_PER_GPU=1
export SERVER_PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
OUT="${OUT:-${ROOT}/outputs/joint_tail_diagnosis_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT}"
echo "Diagnostic output: ${OUT}"
if [[ -z "${TRACE:-}" ]]; then
    if [[ -e "${OUT}/capture" || -e "${OUT}/capture_shards" ]]; then
        echo 'Capture directory already exists; use a fresh OUT or supply TRACE.' >&2; exit 1
    fi
    OPENWAM_REQUEST_TRACE_DIR="${OUT}/capture_shards" OPENWAM_TRACE_WARMUP=0 OPENWAM_TRACE_MEASURE=10000 \
    OUTPUT_DIR="${OUT}/capture" bash scripts/eval_alpha_libero.sh eval \
        --diagnostic --suites spatial --task-ids 0 --num-trials 5 --seed 42 \
        --sampler native --denoise-steps 10 --inference-horizon 8 --inference-mode sync --denoise-mode sync \
        --extra-deploy-arg optimization.dit_cache.enabled=false --extra-deploy-arg optimization.decode_video=false
    "${SERVER_PYTHON}" -m benchmarks.libero.joint_tail_diagnosis assemble \
        --source "${OUT}/capture_shards" --output "${OUT}/native_observations.npz"
    TRACE="${OUT}/native_observations.npz"
fi
EXTRA=()
if [[ -n "${CKPT_NAME:-}" ]]; then EXTRA+=(--ckpt-name "${CKPT_NAME}"); fi
exec "${SERVER_PYTHON}" -m benchmarks.libero.joint_tail_diagnosis replay \
    --trace "${TRACE}" --output "${OUT}/diagnosis" --ckpt-dir "${CKPT_DIR}" \
    --device "${DEVICE:-cuda:${GPU:-0}}" --episode "${EPISODE:-1}" \
    --action-tail-init "${ACTION_TAIL_INIT:-stage_residual}" --video-tail-init "${VIDEO_TAIL_INIT:-stage_residual}" \
    "${EXTRA[@]}"
