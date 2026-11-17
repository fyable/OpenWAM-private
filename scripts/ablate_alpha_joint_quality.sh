#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-diagnostic}"
if [[ $# -gt 1 ]]; then echo 'Configure using environment variables.' >&2; exit 2; fi
case "${MODE}" in plan|diagnostic|summary) ;; *) echo 'Usage: bash scripts/ablate_alpha_joint_quality.sh {plan|diagnostic|summary}' >&2; exit 2 ;; esac
export CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export GPUS="${GPUS:-${GPU:-0}}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
export SERVER_PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
export COMPARE_ROOT="${COMPARE_ROOT:-${ROOT}/outputs/libero/joint_quality_$(date +%Y%m%d_%H%M%S)}"
SUITES="${SUITES:-spatial}"
TASK_IDS="${TASK_IDS:-0}"
METHODS=(phase_shared_n10 phase_unshared_n10 action_unshared_n10 action_shared_n10)
echo "Output: ${COMPARE_ROOT}; suites=${SUITES}; tasks=${TASK_IDS}"
if [[ "${MODE}" == plan ]]; then
    echo '4 methods: phase_shared_n10, phase_unshared_n10, action_unshared_n10, action_shared_n10.'
    echo 'Default: spatial task 0, five initial states, 20 episodes total; seed=42, N10/H32/E8.'
    echo 'One steady-state joint forward; action_only has no cross-request video history. No GPU work in plan mode.'
    exit 0
fi
if [[ "${MODE}" != summary ]]; then
    for name in "${METHODS[@]}"; do
        scope=phase_video_action
        if [[ "${name}" == action_* ]]; then scope=action_only; fi
        shared=false
        if [[ "${name}" == *_shared_* ]]; then shared=true; fi
        OUTPUT_DIR="${COMPARE_ROOT}/${name}" bash scripts/eval_alpha_libero.sh eval \
            --diagnostic --suites "${SUITES}" --task-ids "${TASK_IDS}" --num-trials 5 --seed 42 \
            --sampler joint-paraflow --denoise-steps 10 --inference-horizon 8 \
            --inference-mode sync --denoise-mode sync \
            --extra-deploy-arg optimization.dit_cache.enabled=false \
            --extra-deploy-arg optimization.decode_video=false \
            --extra-deploy-arg inference.joint_cache_schedule=true \
            --extra-deploy-arg "inference.joint_share_conditions=${shared}" \
            --extra-deploy-arg "inference.joint_history_scope=${scope}"
    done
fi
"${SERVER_PYTHON}" -m benchmarks.libero.joint_quality --root "${COMPARE_ROOT}"
