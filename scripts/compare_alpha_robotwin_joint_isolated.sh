#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -f outputs/robotwin/environment.sh ]]; then
    source outputs/robotwin/environment.sh
fi
MODE="${1:-diagnostic}"
PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
if [[ "${MODE}" == replay ]]; then
    : "${TRACE:?Set TRACE to observations.npz produced by capture}"
    EXTRA=()
    if [[ -n "${CKPT_NAME:-}" ]]; then EXTRA+=(--ckpt-name "${CKPT_NAME}"); fi
    if [[ "${ALLOW_PARITY_FAILURE:-0}" == 1 ]]; then EXTRA+=(--allow-parity-failure); fi
    exec "${PYTHON}" -m benchmarks.robotwin.joint_execution_replay \
        --trace "${TRACE}" --output "${REPLAY_ROOT:-${ROOT}/outputs/robotwin/joint_execution_replay_$(date +%Y%m%d_%H%M%S)}" \
        --ckpt-dir "${JOINT_CKPT_DIR:-${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full}}" \
        --device "${DEVICE:-cuda:${GPU:-0}}" --warmup "${WARMUP:-48}" --measure "${MEASURE:-120}" \
        --profile-samples "${PROFILE_SAMPLES:-8}" "${EXTRA[@]}"
fi
if [[ "${MODE}" == execution-all ]]; then
    export COMPARE_ROOT="${COMPARE_ROOT:-${ROOT}/outputs/robotwin/joint_execution_$(date +%Y%m%d_%H%M%S)}"
    CAMPAIGN_ROOT="${COMPARE_ROOT}"
    "${PYTHON}" -m benchmarks.robotwin.compare_joint_isolated execution-ablation
    COMPARE_ROOT="${CAMPAIGN_ROOT}/capture" "${PYTHON}" -m benchmarks.robotwin.compare_joint_isolated capture
    TRACE="${CAMPAIGN_ROOT}/capture/observations.npz" REPLAY_ROOT="${CAMPAIGN_ROOT}/replay" \
        bash "${BASH_SOURCE[0]}" replay
    exit 0
fi
exec "${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}" \
    -m benchmarks.robotwin.compare_joint_isolated "$@"
