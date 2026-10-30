#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CONDA_BIN="${CONDA_BIN:-/home/zhijian/miniconda3/bin/conda}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${REPO_ROOT}/.cache/conda}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.cache/pip}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export SERVER_PYTHON="${SERVER_PYTHON:-${REPO_ROOT}/.venv-openwam/bin/python}"
export LIBERO_ENV_PREFIX="${LIBERO_ENV_PREFIX:-${REPO_ROOT}/.venv-libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_ENV_PREFIX}/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-${REPO_ROOT}/third_party/LIBERO}"
export LIBERO_CONFIG_ROOT="${LIBERO_CONFIG_ROOT:-${REPO_ROOT}/outputs/libero-config}"
export GPUS="${GPUS:-0}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
JOINT_ARGS=()
if [[ "${MODE}" == joint-* ]]; then
    export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/libero/joint_paraflow_${MODE#joint-}_$(date +%Y%m%d_%H%M%S)}"
    JOINT_ARGS=(--sampler joint-paraflow --denoise-steps 4 --inference-horizon 8)
    MODE="${MODE#joint-}"
    if [[ "${MODE}" == diagnostic ]]; then
        MODE=eval
        JOINT_ARGS+=(--diagnostic --num-trials 5 --task-ids 0)
    fi
fi
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO}"
CKPT_NAME="${CKPT_NAME:-checkpoint_step_10690.safetensors}"

case "${MODE}" in
    setup)
        [[ -x "${CONDA_BIN}" ]] || { echo "Conda not found: ${CONDA_BIN}" >&2; exit 1; }
        if [[ ! -x "${SERVER_PYTHON}" ]]; then
            "${CONDA_BIN}" --no-plugins create --solver classic -y \
                --prefix "$(dirname "$(dirname "${SERVER_PYTHON}")")" python=3.10 pip
        fi
        "${SERVER_PYTHON}" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
            --index-url https://download.pytorch.org/whl/cu128
        "${SERVER_PYTHON}" -m pip install --editable "${REPO_ROOT}"
        bash benchmarks/libero/setup_env.sh
        bash benchmarks/libero/run_smoke.sh task
        ;;
    download)
        "${SERVER_PYTHON}" scripts/download_assets/download_openwam_checkpoints.py \
            --family alpha --name OpenWAM-Alpha-Sim-LIBERO --yes "$@"
        ;;
    env)
        bash benchmarks/libero/run_smoke.sh env "$@"
        ;;
    smoke|eval)
        [[ -x "${SERVER_PYTHON}" && -x "${LIBERO_PYTHON}" ]] || {
            echo "Environments missing; run: bash scripts/eval_alpha_libero.sh setup" >&2
            exit 1
        }
        for required_file in config.yaml normalization_stats.npy "${CKPT_NAME}"; do
            [[ -f "${CKPT_DIR}/${required_file}" ]] || {
                echo "Checkpoint file missing: ${CKPT_DIR}/${required_file}" >&2
                exit 1
            }
        done
        "${SERVER_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; check the NVIDIA driver and GPU access"'
        EXTRA_ARGS=()
        if [[ "${MODE}" == smoke ]]; then EXTRA_ARGS+=(--smoke); fi
        if [[ -n "${RENDER_GPUS:-}" ]]; then EXTRA_ARGS+=(--render-gpus "${RENDER_GPUS}"); fi
        bash benchmarks/libero/run_eval.sh "${CKPT_DIR}" "${CKPT_NAME}" \
            --compile-enabled false "${JOINT_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"
        ;;
    help|-h|--help)
        echo "Usage: bash scripts/eval_alpha_libero.sh {setup|download|env|smoke|eval|joint-smoke|joint-diagnostic|joint-eval} [options]"
        echo "Joint modes use N4/E8 and the OpenWAM-local joint solver."
        echo "Overrides: CONDA_BIN, SERVER_PYTHON, LIBERO_ENV_PREFIX, LIBERO_PYTHON, LIBERO_PATH"
        echo "           CKPT_DIR, CKPT_NAME, GPUS, RENDER_GPUS, REPLICAS_PER_GPU, OUTPUT_DIR"
        ;;
    *)
        echo "Unknown mode: ${MODE}; use --help" >&2
        exit 2
        ;;
esac
