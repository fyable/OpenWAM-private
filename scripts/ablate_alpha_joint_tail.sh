#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-diagnostic}"
if [[ $# -gt 1 ]]; then echo 'Configure using environment variables.' >&2; exit 2; fi
case "${MODE}" in plan|diagnostic|summary) ;; *) echo 'Usage: bash scripts/ablate_alpha_joint_tail.sh {plan|diagnostic|summary}' >&2; exit 2 ;; esac
export CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export GPUS="${GPUS:-${GPU:-0}}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
export SERVER_PYTHON="${SERVER_PYTHON:-${ROOT}/.venv-openwam/bin/python}"
export COMPARE_ROOT="${COMPARE_ROOT:-${ROOT}/outputs/libero/joint_tail_$(date +%Y%m%d_%H%M%S)}"
echo "Output: ${COMPARE_ROOT}"
if [[ "${MODE}" == plan ]]; then
    echo 'Four tail combinations: residual/residual, bridge/residual, residual/bridge, bridge/bridge.'
    echo 'Default spatial task 0, five initial states, 20 episodes total; N10/H32/E8, one warm forward.'
    exit 0
fi
if [[ "${MODE}" != summary ]]; then
    for name in tail_residual_n10 tail_action_bridge_n10 tail_video_bridge_n10 tail_both_bridge_n10; do
        action=stage_residual
        video=stage_residual
        case "${name}" in tail_action_bridge_n10|tail_both_bridge_n10) action=endpoint_bridge ;; esac
        case "${name}" in tail_video_bridge_n10|tail_both_bridge_n10) video=endpoint_bridge ;; esac
        OUTPUT_DIR="${COMPARE_ROOT}/${name}" bash scripts/eval_alpha_libero.sh eval \
            --diagnostic --suites "${SUITES:-spatial}" --task-ids "${TASK_IDS:-0}" --num-trials 5 --seed 42 \
            --sampler joint-paraflow --denoise-steps 10 --inference-horizon 8 \
            --inference-mode sync --denoise-mode sync \
            --extra-deploy-arg optimization.dit_cache.enabled=false \
            --extra-deploy-arg optimization.decode_video=false \
            --extra-deploy-arg inference.joint_cache_schedule=true \
            --extra-deploy-arg inference.joint_share_conditions=true \
            --extra-deploy-arg inference.joint_history_scope=phase_video_action \
            --extra-deploy-arg "inference.joint_action_tail_init=${action}" \
            --extra-deploy-arg "inference.joint_video_tail_init=${video}"
    done
fi
"${SERVER_PYTHON}" -m benchmarks.libero.joint_quality --root "${COMPARE_ROOT}" --preset tail
