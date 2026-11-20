#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")/.."
root=$1
role=$2
gpu=$3
deadline=$4
export LIBERO_PATH="$PWD/third_party/LIBERO" PYTHONPATH="$PWD:$PWD/third_party/LIBERO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export E2_MEMORY_MODE=text-offload N10_RECORD_LATENCY=1
export GPUS="$gpu" BASE_PORT="$((8950 + gpu))" LIBERO_PYTHON="$PWD/.venv-openwam/bin/python"
export SERVER_PYTHON="$PWD/scripts/native_n10_refresh_python.sh"
export CKPT_DIR="$PWD/.cache/models/OpenWAM-Alpha-Sim-LIBERO" CKPT_NAME=checkpoint_step_10690.safetensors
export LIBERO_CONFIG_ROOT="$root/libero-gpu$gpu" LIBERO_CONFIG_PATH="$root/libero-gpu$gpu"
trap 'printf "%s\n" "$?" > "$root/$role.exit"' EXIT
if [[ "$role" == graph-* ]]; then
    export N10_ACTION_GRAPH=1 N10_REFRESH_STEPS=0
else
    export N10_ACTION_GRAPH=0 N10_REFRESH_STEPS=''
fi
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
exec 9>>".cache/e2-reproduction-$uuid.lock"
flock -n 9
if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | rg -q '^[0-9]+$'; then
    exit 1
fi
run_eval() {
    local phase=$1 suites=$2 tasks=$3
    local remaining=$(( $(date -d "$deadline" +%s) - $(date +%s) ))
    (( remaining > 0 )) || return 124
    export OUTPUT_DIR="$root/$role-$phase"
    timeout -k 30 "$remaining" bash scripts/eval_alpha_libero.sh eval --sampler native \
        --denoise-steps 10 --inference-horizon 8 --inference-mode sync --denoise-mode sync \
        --extra-deploy-arg inference.cfg_scale=1.0 --extra-deploy-arg optimization.dit_cache.enabled=false \
        --extra-deploy-arg optimization.decode_video=false --extra-deploy-arg inference.request_timing=true \
        --diagnostic --suites "$suites" --task-ids "$tasks" --num-trials 5 --seed 42
}
case "$role" in
    *-smoke) run_eval smoke spatial,goal 0 ;;
    *-main) run_eval main spatial,goal,object,long 2,5,8 ;;
    *) exit 2 ;;
esac
