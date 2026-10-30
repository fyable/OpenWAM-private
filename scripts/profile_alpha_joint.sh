#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "${MODE}" in all|capture|plan|timing|profile|summary) ;; *) echo 'Usage: bash scripts/profile_alpha_joint.sh {all|capture|plan|timing|profile|summary}' >&2; exit 2 ;; esac
BENCHMARK="${BENCHMARK:-libero}"
GPU="${GPU:-0}"
PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
OUT="${PROFILE_ROOT:-${ROOT}/outputs/joint_profile_$(date +%Y%m%d_%H%M%S)}"
case "${BENCHMARK}" in
    libero) DEFAULT_CKPT=/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO ;;
    robotwin) DEFAULT_CKPT=/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full ;;
    *) echo 'BENCHMARK must be libero or robotwin' >&2; exit 2 ;;
esac
COMMON=(--output "${OUT}" --benchmark "${BENCHMARK}" --ckpt-dir "${CKPT_DIR:-${DEFAULT_CKPT}}"
    --device "${DEVICE:-cuda:${GPU}}" --gpu "${GPU}" --port "${PORT:-8950}"
    --warmup "${WARMUP:-48}" --measure "${MEASURE:-120}" --suite "${SUITE:-spatial}"
    --micro-warmup "${MICRO_WARMUP:-5}" --micro-measure "${MICRO_MEASURE:-20}"
    --task-id "${TASK_ID:-0}" --capture-timeout "${CAPTURE_TIMEOUT:-7200}")
if [[ -n "${TRACE:-}" ]]; then COMMON+=(--trace "${TRACE}"); fi
if [[ -n "${CKPT_NAME:-}" ]]; then COMMON+=(--ckpt-name "${CKPT_NAME}"); fi
if [[ -n "${CAPTURE_COMMAND_JSON:-}" ]]; then COMMON+=(--capture-command-json "${CAPTURE_COMMAND_JSON}"); fi
if [[ "${SKIP_PROFILE:-0}" == 1 ]]; then COMMON+=(--skip-profile); fi
if [[ "${ALLOW_PARITY_FAILURE:-0}" == 1 ]]; then COMMON+=(--allow-parity-failure); fi
if [[ "${MODE}" != plan ]]; then
    echo "Output: ${OUT}; GPU=${GPU}, device=${DEVICE:-cuda:${GPU}}. Select an available GPU; existing jobs are never stopped."
fi
exec "${PYTHON}" -m benchmarks.libero.joint_experiment "${MODE}" "${COMMON[@]}" "$@"
