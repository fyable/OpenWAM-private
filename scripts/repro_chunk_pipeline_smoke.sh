#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
CKPT_NAME=checkpoint_step_10690.safetensors
GPU_LIST="${GPU_LIST:-1,2,3,0}"
HEAD_GPU="${GPU_LIST%%,*}"
BASE_PORT="${BASE_PORT:-8993}"
RUN="${RUN:-$REPO_ROOT/experiment/chunk_pipeline_repro_$(date +%Y%m%d_%H%M%S)}"
export SERVER_PYTHON="${SERVER_PYTHON:-$REPO_ROOT/.venv-openwam/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$REPO_ROOT/.venv-libero/bin/python}"
export LIBERO_PATH="$REPO_ROOT/third_party/LIBERO"
export PYTHONPATH="$REPO_ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export REPLICAS_PER_GPU=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENWAM_LIBERO_REQUEST_TIMING=1
export OPENWAM_REQUIRE_GPU_RESIDENT=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset E2_MEMORY_MODE
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
printf "run=%s\ngpus=%s\ncheckpoint=%s/%s\nmemory_policy=gpu-resident-no-offload\n" \
  "$RUN" "$GPU_LIST" "$CKPT_DIR" "$CKPT_NAME"
cp scripts/repro_chunk_pipeline_smoke.sh "$RUN/run.sh"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee "$RUN/gpus-before.csv"
sha256sum openwam/deploy/{chunk_pipeline.py,engine.py,server.py} \
  openwam/deploy/model_loader.py benchmarks/libero/{scheduler.py,policy_config.yml,openwam2libero_interface.py} \
  scripts/{repro_chunk_pipeline_smoke.sh,summarize_chunk_pipeline_smoke.py} | tee "$RUN/source-sha256.txt"
printf "%s  %s/%s\n" \
  34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a \
  "$CKPT_DIR" "$CKPT_NAME" | sha256sum -c -
test -f "$CKPT_DIR/config.yaml"
test -f "$CKPT_DIR/normalization_stats.npy"
test -d "$CKPT_DIR/tokenizer"
"$SERVER_PYTHON" -c "import sys, torch; print(sys.version, torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.device_count()); devices = [int(value) for value in sys.argv[1].split(\",\")]; assert len(devices) == len(set(devices)) == 4; assert all(0 <= device < torch.cuda.device_count() for device in devices)" "$GPU_LIST"
"$LIBERO_PYTHON" -c "import sys, mujoco, robosuite; print(sys.version, mujoco.__version__, robosuite.__version__)"
"$SERVER_PYTHON" -c "import socket; connection = socket.socket(); connection.bind((\"127.0.0.1\", int(\"$BASE_PORT\"))); connection.close()"

common=(--denoise-steps 10 --inference-horizon 8 --inference-mode sync
  --denoise-mode sync --compile-enabled false --seed 42
  --suites spatial,goal,object,long --render-gpus "$HEAD_GPU"
  --extra-deploy-arg optimization.dit_cache.enabled=false
  --extra-deploy-arg optimization.decode_video=false
  --extra-deploy-arg inference.request_timing=true
  --diagnostic --task-ids 0 --num-trials 5 --client-max-attempts 1)

for sampler in native chunk-pipeline; do
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv \
    > "$RUN/gpu-processes-before-$sampler.csv"
  gpus="$HEAD_GPU"
  extra=()
  if [[ "$sampler" == chunk-pipeline ]]; then
    gpus="$GPU_LIST"
    extra=(--chunk-speculative-steps 3)
  fi
  GPUS="$gpus" BASE_PORT="$BASE_PORT" OUTPUT_DIR="$RUN/smoke/$sampler" \
    bash benchmarks/libero/run_eval.sh "$CKPT_DIR" "$CKPT_NAME" \
      --sampler "$sampler" "${common[@]}" "${extra[@]}"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv \
    > "$RUN/gpu-processes-after-$sampler.csv"
done
"$SERVER_PYTHON" scripts/summarize_chunk_pipeline_smoke.py "$RUN"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee "$RUN/gpus-after.csv"
