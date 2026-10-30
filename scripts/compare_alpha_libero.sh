#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
MODE="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then echo "Configure this script with environment variables, not extra arguments." >&2; exit 2; fi
case "${MODE}" in
    full|diagnostic|plan|summary|replay) ;;
    *) echo "Usage: bash scripts/compare_alpha_libero.sh {full|diagnostic|plan|summary|replay}"; exit 2 ;;
esac
export CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export GPUS="${GPUS:-0}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
export SERVER_PYTHON="${SERVER_PYTHON:-${REPO_ROOT}/.venv-openwam/bin/python}"
export COMPARE_ROOT="${COMPARE_ROOT:-${REPO_ROOT}/outputs/libero/compare_joint_$(date +%Y%m%d_%H%M%S)}"
echo "Comparison directory: ${COMPARE_ROOT}"
COMMON=(--suites spatial,goal,object,long --task-ids all --num-trials 50 --seed 42
    --inference-mode sync --denoise-mode sync --inference-horizon 8
    --extra-deploy-arg optimization.dit_cache.enabled=false
    --extra-deploy-arg optimization.decode_video=false)
if [[ "${MODE}" == diagnostic ]]; then
    COMMON+=(--diagnostic --task-ids 0 --num-trials 5)
fi
if [[ "${MODE}" == plan ]]; then
    echo "Full: 3 methods x 40 tasks x 50 trials = 6000 episodes; sequential execution."
    echo "Methods: native_n10_e8, native_n4_e8, joint_n4_e8; compile/cache/decode off."
    echo "Use diagnostic for 20 episodes per method; summary never starts evaluation."
    echo "Resume: COMPARE_ROOT=/existing/directory bash scripts/compare_alpha_libero.sh full"
    exit 0
fi
if [[ "${MODE}" == replay ]]; then
    : "${TRACE:?Set TRACE to an E8 observation NPZ; see JOINT_PARAFLOW.md}"
    mkdir -p "${COMPARE_ROOT}/replay"
    "${SERVER_PYTHON}" -m benchmarks.libero.replay_joint --observations "${TRACE}" \
        --ckpt-dir "${CKPT_DIR}" --sampler joint-paraflow --steps 4 --verify-batch \
        --output "${COMPARE_ROOT}/replay/parity.json"
    "${SERVER_PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["stage_batch_parity"]["status"] == "passed", "No verified warm batch"' "${COMPARE_ROOT}/replay/parity.json"
    for spec in native:10 native:4 joint-paraflow:4; do
        method="${spec%:*}"; steps="${spec#*:}"
        "${SERVER_PYTHON}" -m benchmarks.libero.replay_joint --observations "${TRACE}" \
            --ckpt-dir "${CKPT_DIR}" --sampler "${method}" --steps "${steps}" \
            --output "${COMPARE_ROOT}/replay/${method}_n${steps}.json"
    done
    exit 0
fi
if [[ "${MODE}" != summary ]]; then
    mkdir -p "${COMPARE_ROOT}"
    for spec in native_n10_e8:10:native native_n4_e8:4:native joint_n4_e8:4:joint-paraflow; do
        IFS=: read -r name steps sampler <<< "${spec}"
        OUTPUT_DIR="${COMPARE_ROOT}/${name}" bash scripts/eval_alpha_libero.sh eval \
            --sampler "${sampler}" --denoise-steps "${steps}" "${COMMON[@]}"
    done
fi
"${SERVER_PYTHON}" -m benchmarks.libero.compare_joint --root "${COMPARE_ROOT}"
