#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -f outputs/robotwin/environment.sh && "${BENCHMARK:-libero}" == robotwin ]]; then
    source outputs/robotwin/environment.sh
fi
PYTHON="${VIDEO_PARALLEL_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
if [[ "${WAIT_FOR_GPU:-0}" == 1 && " $* " != *" --summary "* ]]; then
    exec "${PYTHON}" -m benchmarks.video_parallel.gpu_wait \
        --gpus "${GPUS:-${GPU:-auto}}" --poll "${GPU_POLL_SECONDS:-30}" \
        --idle-seconds "${GPU_IDLE_SECONDS:-60}" --timeout "${GPU_WAIT_TIMEOUT:-0}" \
        --max-memory-mib "${GPU_MAX_MEMORY_MIB:-1024}" --max-utilization "${GPU_MAX_UTILIZATION:-5}" \
        -- bash "${ROOT}/scripts/evaluate_video_parallel.sh" "$@"
fi
ARGS=(--benchmark "${BENCHMARK:-libero}" --stage "${STAGE:-screen}"
    --gpus "${GPUS:-${GPU:-0}}" --output "${OUT:-${ROOT}/outputs/video_parallel_closed_loop_$(date +%Y%m%d_%H%M%S)}")
if [[ -n "${CKPT_DIR:-}" ]]; then ARGS+=(--ckpt-dir "${CKPT_DIR}"); fi
if [[ -n "${METHODS:-}" ]]; then ARGS+=(--methods "${METHODS}"); fi
if [[ -n "${TASKS:-}" ]]; then ARGS+=(--tasks "${TASKS}"); fi
if [[ -n "${SOLVER_OVERRIDES:-}" ]]; then ARGS+=(--overrides "${SOLVER_OVERRIDES}"); fi
exec "${PYTHON}" -m benchmarks.video_parallel.closed_loop "${ARGS[@]}" "$@"
