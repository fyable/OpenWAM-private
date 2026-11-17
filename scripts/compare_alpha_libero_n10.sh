#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-full}"
if [[ $# -gt 1 ]]; then echo 'Configure with environment variables, not extra arguments.' >&2; exit 2; fi
case "${MODE}" in full|diagnostic|plan|summary) ;; *) echo 'Usage: bash scripts/compare_alpha_libero_n10.sh {full|diagnostic|plan|summary}' >&2; exit 2 ;; esac
export CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export GPUS="${GPUS:-${GPU:-0}}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
export SERVER_PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
export COMPARE_ROOT="${COMPARE_ROOT:-${ROOT}/outputs/libero/compare_n10_$(date +%Y%m%d_%H%M%S)}"
echo "Comparison directory: ${COMPARE_ROOT}"
if [[ "${MODE}" == plan ]]; then
    echo 'Native Joint N10 versus optimized parallel Joint N10 (schedule cache + shared conditions).'
    echo 'Full: 2 methods x 40 tasks x 50 trials = 4000 episodes; diagnostic: 40 episodes total.'
    echo 'Same checkpoint, seed=42, H32/E8, mutual attention; video latent denoising on, pixel decode off.'
    echo 'Sequential closed-loop evaluation; no parity gate bypass or success-rate guarantee.'
    exit 0
fi
if [[ "${MODE}" != summary ]]; then
    COMMON=(--suites spatial,goal,object,long --task-ids all --num-trials 50 --seed 42
        --inference-mode sync --denoise-mode sync --inference-horizon 8 --denoise-steps 10
        --extra-deploy-arg optimization.dit_cache.enabled=false
        --extra-deploy-arg optimization.decode_video=false)
    if [[ "${MODE}" == diagnostic ]]; then
        COMMON+=(--diagnostic --suites "${SUITES:-spatial,goal,object,long}" --task-ids "${TASK_IDS:-0}" --num-trials 5)
    fi
    for spec in native_n10_e8:native joint_shared_n10_e8:joint-paraflow; do
        IFS=: read -r name sampler <<< "${spec}"
        enabled=false
        if [[ "${sampler}" == joint-paraflow ]]; then enabled=true; fi
        OUTPUT_DIR="${COMPARE_ROOT}/${name}" bash scripts/eval_alpha_libero.sh eval \
            --sampler "${sampler}" "${COMMON[@]}" \
            --extra-deploy-arg "inference.joint_cache_schedule=${enabled}" \
            --extra-deploy-arg "inference.joint_share_conditions=${enabled}"
    done
fi
"${SERVER_PYTHON}" -m benchmarks.libero.compare_joint --root "${COMPARE_ROOT}" \
    --methods native_n10_e8 joint_shared_n10_e8
