#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU_LIST="${GPU_LIST:-0,2,4,6}"
HEAD_GPU="${GPU_LIST%%,*}"
CKPT_DIR="${CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
CKPT_NAME=checkpoint_step_10690.safetensors
RUN="${RUN:-$REPO_ROOT/experiment/chunk_latency_$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-8993}"
export SERVER_PYTHON="${SERVER_PYTHON:-$REPO_ROOT/.venv-openwam/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$REPO_ROOT/.venv-libero/bin/python}"
export LIBERO_PATH="$REPO_ROOT/third_party/LIBERO"
export PYTHONPATH="$REPO_ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export REPLICAS_PER_GPU=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENWAM_LIBERO_REQUEST_TIMING=1 OPENWAM_REQUIRE_GPU_RESIDENT=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset E2_MEMORY_MODE
mkdir -p "$RUN"
if [[ -f "$RUN/source-sha256.txt" ]]; then
  RECOVERY_TAG="$(date +%Y%m%d_%H%M%S)"
  cp "$RUN/source-sha256.txt" "$RUN/source-sha256-before-$RECOVERY_TAG.txt"
  cp "$RUN/run.sh" "$RUN/run-before-$RECOVERY_TAG.sh"
  cp "$RUN/gpus-before.csv" "$RUN/gpus-before-$RECOVERY_TAG.csv"
fi
exec > >(tee -a "$RUN/run.log") 2>&1
cp scripts/profile_chunk_pipeline.sh "$RUN/run.sh"
printf 'run=%s\ngpus=%s\nalgorithm=round-barrier-N10-H32-E8-budget3\n' "$RUN" "$GPU_LIST"
"$SERVER_PYTHON" -c 'import json,time; print(json.dumps({"wall_time_s":time.time(),"perf_counter_s":time.perf_counter()}))' > "$RUN/clock-anchor.json"
sha256sum openwam/chunk_diagnostics.py openwam/deploy/{chunk_pipeline.py,engine.py,server.py,model_loader.py,executors/sync_executor.py} \
  openwam/model/video_backbone/wan/{preprocess,encode,conditioning}.py \
  benchmarks/libero/{single_eval.py,openwam2libero_interface.py,policy_config.yml,scheduler.py} \
  scripts/{profile_chunk_pipeline.sh,probe_chunk_profiler.py,summarize_chunk_diagnostics.py} > "$RUN/source-sha256.txt"
printf '%s  %s/%s\n' 34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a "$CKPT_DIR" "$CKPT_NAME" | sha256sum -c -
"$SERVER_PYTHON" -c 'import sys, torch; print(sys.version, torch.__version__); cards=[int(value) for value in sys.argv[1].split(",")]; assert len(set(cards)) == len(cards) == 4; assert all(0 <= card < torch.cuda.device_count() for card in cards)' "$GPU_LIST"
"$LIBERO_PYTHON" -c 'import sys, mujoco, robosuite; print(sys.version, mujoco.__version__, robosuite.__version__)'
"$SERVER_PYTHON" -c 'import socket,sys; connection=socket.socket(); connection.bind(("127.0.0.1",int(sys.argv[1]))); connection.close()' "$BASE_PORT"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv > "$RUN/gpus-before.csv"
monitor() {
  while true; do
    date --iso-8601=ns >> "$RUN/gpu-util.csv"
    nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader >> "$RUN/gpu-util.csv" || true
    date --iso-8601=ns >> "$RUN/gpu-processes.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$RUN/gpu-processes.csv" || true
    sleep 1
  done
}
monitor &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$SERVER_PYTHON" scripts/probe_chunk_profiler.py "$RUN/probe"
common=(--sampler chunk-pipeline --chunk-speculative-steps 3
  --denoise-steps 10 --inference-horizon 8 --inference-mode sync --denoise-mode sync
  --compile-enabled false --seed 42 --render-gpus "$HEAD_GPU"
  --extra-deploy-arg optimization.dit_cache.enabled=false
  --extra-deploy-arg optimization.decode_video=false
  --extra-deploy-arg inference.request_timing=true
  --diagnostic --task-ids 0 --client-max-attempts 1)
for phase in ${PHASES:-control segments trace}; do
  export OPENWAM_CHUNK_DIAGNOSTICS=0 OPENWAM_CHUNK_TRACE=0
  export OPENWAM_CHUNK_DIAGNOSTICS_DIR="$RUN/$phase/diagnostics"
  extra=(--suites spatial,goal,object,long --num-trials 5)
  if [[ "$phase" != control ]]; then export OPENWAM_CHUNK_DIAGNOSTICS=1; fi
  if [[ "$phase" == trace ]]; then
    if ! "$SERVER_PYTHON" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["complete"])' "$RUN/probe/probe.json"; then
      echo 'Detailed trace skipped: four-thread/four-device capture probe incomplete.'
      continue
    fi
    export OPENWAM_CHUNK_TRACE=1
    extra=(--suites spatial --num-trials 1 --smoke)
  fi
  printf 'phase=%s start=%s\n' "$phase" "$(date --iso-8601=ns)"
  GPUS="$GPU_LIST" BASE_PORT="$BASE_PORT" OUTPUT_DIR="$RUN/$phase" \
    bash benchmarks/libero/run_eval.sh "$CKPT_DIR" "$CKPT_NAME" "${common[@]}" "${extra[@]}"
  printf 'phase=%s end=%s\n' "$phase" "$(date --iso-8601=ns)"
done
"$SERVER_PYTHON" scripts/summarize_chunk_diagnostics.py "$RUN"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv > "$RUN/gpus-after.csv"
