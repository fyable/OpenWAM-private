#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
if [[ "${WAIT_FOR_GPU:-0}" == 1 && "${MODE}" != preflight && "${MODE}" != report ]]; then
    exec "${PYTHON}" -m benchmarks.video_parallel.gpu_wait \
        --gpus "${GPU:-auto}" --poll "${GPU_POLL_SECONDS:-30}" \
        --idle-seconds "${GPU_IDLE_SECONDS:-60}" --timeout "${GPU_WAIT_TIMEOUT:-0}" \
        --max-memory-mib "${GPU_MAX_MEMORY_MIB:-1024}" --max-utilization "${GPU_MAX_UTILIZATION:-5}" \
        -- bash "${ROOT}/scripts/profile_video_parallel.sh" "${MODE}" "$@"
fi
OUT="${OUT:-${ROOT}/outputs/video_parallel_$(date +%Y%m%d_%H%M%S)}"
ARGS=(--output "${OUT}")
if [[ "${MODE}" != report ]]; then
    : "${TRACE:?Set TRACE to a real observations.npz}"
    ARGS+=(--trace "${TRACE}" --ckpt-dir "${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
        --device "${DEVICE:-cuda:${GPU:-0}}" --warmup "${WARMUP:-10}" --measure "${MEASURE:-30}"
        --validation-samples "${VALIDATION_SAMPLES:-3}" --min-free-gib "${MIN_FREE_GIB:-5}")
    if [[ -n "${CKPT_NAME:-}" ]]; then ARGS+=(--ckpt-name "${CKPT_NAME}"); fi
    if [[ -n "${METHODS:-}" ]]; then ARGS+=(--methods "${METHODS}"); fi
    if [[ -n "${SOLVER_OVERRIDES:-}" ]]; then ARGS+=(--solver-overrides "${SOLVER_OVERRIDES}"); fi
    if [[ "${ALLOW_PARITY_FAILURE:-0}" == 1 ]]; then ARGS+=(--allow-parity-failure); fi
    if [[ "${ALLOW_UNVERIFIED_TRACE:-0}" == 1 ]]; then ARGS+=(--allow-unverified-trace); fi
    if [[ "${DECODE_VIDEO:-0}" == 1 ]]; then ARGS+=(--decode-video); fi
fi
exec "${PYTHON}" -m benchmarks.video_parallel.experiment "${MODE}" "${ARGS[@]}" "$@"
