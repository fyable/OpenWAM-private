#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-diagnostic}"
if [[ $# -gt 1 ]]; then echo 'Configure with environment variables.' >&2; exit 2; fi
case "${MODE}" in plan|preflight|diagnostic|summary|replay) ;; *) echo 'Usage: bash scripts/compare_alpha_libero_isolated.sh {plan|preflight|diagnostic|summary|replay}' >&2; exit 2 ;; esac
export CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export GPUS="${GPUS:-${GPU:-0}}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
export SERVER_PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
export COMPARE_ROOT="${COMPARE_ROOT:-${ROOT}/outputs/libero/isolated_ablation_$(date +%Y%m%d_%H%M%S)}"
PF_ROOT="${PF_ROOT:-/home/zhijian/paraflow-main}"
PF_NATIVE_ROOT="${PF_NATIVE_ROOT:-/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port/native/openwam}"
echo "Output: ${COMPARE_ROOT}"
if [[ "${MODE}" == plan ]]; then
    echo 'LIBERO checkpoint, experimental isolated override; NOT original isolated weights.'
    echo 'Mutual Native N10, isolated Native N10, isolated original ParaFlow N10.'
    echo 'Default spatial task 0: five states x three methods = 15 episodes, seed42 H32/E8.'
    echo 'Isolated methods use first-frame prefill and no future video; mutual includes future video.'
    exit 0
fi
COMMON_REF=(--paraflow-root "${PF_ROOT}" --native-root "${PF_NATIVE_ROOT}")
if [[ "${MODE}" == preflight ]]; then
    exec "${SERVER_PYTHON}" -m benchmarks.libero.isolated_experiment preflight "${COMMON_REF[@]}"
fi
if [[ "${MODE}" == replay ]]; then
    : "${TRACE:?Set TRACE to real E8 observations}"
    EXTRA=()
    if [[ -n "${CKPT_NAME:-}" ]]; then EXTRA+=(--ckpt-name "${CKPT_NAME}"); fi
    if [[ "${ALLOW_PARITY_FAILURE:-0}" == 1 ]]; then EXTRA+=(--allow-parity-failure); fi
    exec "${SERVER_PYTHON}" -m benchmarks.libero.isolated_experiment replay "${COMMON_REF[@]}" \
        --output "${COMPARE_ROOT}/replay" --trace "${TRACE}" --ckpt-dir "${CKPT_DIR}" \
        --device "${DEVICE:-cuda:${GPU:-0}}" --warmup "${WARMUP:-48}" --measure "${MEASURE:-120}" "${EXTRA[@]}"
fi
if [[ "${MODE}" == diagnostic ]]; then
    "${SERVER_PYTHON}" -m benchmarks.libero.isolated_experiment preflight "${COMMON_REF[@]}"
    for spec in mutual_native_n10:native isolated_native_n10:isolated-native isolated_paraflow_n10:isolated-paraflow; do
        IFS=: read -r name sampler <<< "${spec}"
        EXTRA=()
        if [[ "${sampler}" == isolated-* ]]; then
            ALLOW=false
            if [[ "${ALLOW_PARITY_FAILURE:-0}" == 1 ]]; then ALLOW=true; fi
            EXTRA+=(--extra-deploy-arg "inference.isolated_paraflow_root=${PF_ROOT}"
                --extra-deploy-arg "inference.isolated_native_root=${PF_NATIVE_ROOT}"
                --extra-deploy-arg "inference.isolated_allow_parity_failure=${ALLOW}")
        fi
        OUTPUT_DIR="${COMPARE_ROOT}/${name}" bash scripts/eval_alpha_libero.sh eval \
            --diagnostic --suites "${SUITES:-spatial}" --task-ids "${TASK_IDS:-0}" --num-trials 5 --seed 42 \
            --sampler "${sampler}" --denoise-steps 10 --inference-horizon 8 --inference-mode sync --denoise-mode sync \
            --extra-deploy-arg optimization.dit_cache.enabled=false --extra-deploy-arg optimization.decode_video=false "${EXTRA[@]}"
    done
fi
exec "${SERVER_PYTHON}" -m benchmarks.libero.isolated_experiment summary --output "${COMPARE_ROOT}"
