#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export GPUS=0,2,4,6 REPLICAS_PER_GPU=1
export SERVER_PYTHON="$ROOT/.venv-openwam/bin/python"
export LIBERO_PYTHON="$ROOT/.venv-libero/bin/python"
export LIBERO_PATH="$ROOT/third_party/LIBERO"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENWAM_CHUNK_INTEROP_THREADS=1 OPENWAM_REQUIRE_GPU_RESIDENT=1
export OPENWAM_CHUNK_CPU_GROUPS='0-7;8-15;16-23;24-31'
export OPENWAM_CHUNK_SELECTED_FRAMES=1 OPENWAM_CHUNK_REUSE_NOISE=1
export OPENWAM_CHUNK_TIMESTEP_CACHE=1 OPENWAM_CHUNK_CUDA_GRAPH=1
export OPENWAM_CHUNK_VAE_GPU_ACCUM=1 OPENWAM_CHUNK_VAE_GRAPH=1
export OPENWAM_LIBERO_REQUEST_TIMING=1 OPENWAM_LIBERO_STATE_HASH=1
export OPENWAM_CHUNK_TRACE=0 OPENWAM_CHUNK_DIAGNOSTICS=0
unset E2_MEMORY_MODE OPENWAM_CHUNK_CAPTURE_PATH OPENWAM_CHUNK_SHADOW_PARITY
export BASE_PORT=${BASE_PORT:-9015}
export OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/experiment/chunk_graph_repro_$(date +%Y%m%d_%H%M%S)"}

bash benchmarks/libero/run_eval.sh \
  "${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}" \
  checkpoint_step_10690.safetensors \
  --sampler chunk-pipeline --chunk-speculative-steps 3 \
  --denoise-steps 10 --inference-horizon 8 --inference-mode sync --denoise-mode sync \
  --compile-enabled false --seed 42 --render-gpus 0 --server-start-timeout 1800 \
  --client-max-attempts 1 --diagnostic --task-ids "${TASK_IDS:-0}" --num-trials 5 \
  --suites spatial,goal,object,long \
  --extra-deploy-arg optimization.dit_cache.enabled=false \
  --extra-deploy-arg optimization.decode_video=false \
  --extra-deploy-arg inference.request_timing=true \
  --extra-deploy-arg inference.chunk_worker_backend=process
