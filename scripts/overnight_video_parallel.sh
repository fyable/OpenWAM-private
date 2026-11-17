#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export TRACE="${TRACE:-${ROOT}/outputs/joint_profile_20261029_231836/observations.npz}"
QUEUE_OUT="${QUEUE_OUT:-${ROOT}/outputs/video_parallel_overnight_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${QUEUE_OUT}"
exec 9>"${QUEUE_OUT}/queue.lock"
flock -n 9 || { echo "This queue is already running: ${QUEUE_OUT}"; exit 1; }
exec > >(tee -a "${QUEUE_OUT}/queue.log") 2>&1
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    echo "Unset CUDA_VISIBLE_DEVICES; GPU uses physical indices."
    exit 1
fi
[[ -f "${TRACE}" ]] || { echo "Trace missing: ${TRACE}"; exit 1; }
export GPU="${NIGHT_GPU:-auto}" WAIT_FOR_GPU=1
export CKPT_DIR="${NIGHT_CKPT_DIR:-/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO}"
export CKPT_NAME="${NIGHT_CKPT_NAME:-}"
export ALLOW_UNVERIFIED_TRACE="${ALLOW_UNVERIFIED_TRACE:-1}"
export ALLOW_PARITY_FAILURE="${ALLOW_PARITY_FAILURE:-0}"
export DECODE_VIDEO=0 SOLVER_OVERRIDES='{}'
printf 'Queue: %s\nTrace: %s\nGPU: %s\n' "${QUEUE_OUT}" "${TRACE}" "${GPU}"
printf 'Checkpoint: %s (name=%s)\n' "${CKPT_DIR}" "${CKPT_NAME:-auto}"
echo 'Historical unverified trace: diagnostic results only. No closed-loop or paper-quality claims.'
echo 'Waiting time is unlimited by default. Ctrl-C stops the queue, not other GPU jobs.'
printf 'time\tjob\texit_code\n' >> "${QUEUE_OUT}/status.tsv"
run_job() {
    local name="$1" mode="$2" code=0
    shift 2
    echo "[$(date -Is)] START ${name} (includes waiting for GPU)"
    OUT="${QUEUE_OUT}/${name}" bash scripts/profile_video_parallel.sh "${mode}" "$@" || code=$?
    printf '%s\t%s\t%s\n' "$(date -Is)" "${name}" "${code}" >> "${QUEUE_OUT}/status.tsv"
    echo "[$(date -Is)] END ${name}: ${code}"
    if [[ "${code}" == 130 || "${code}" == 143 ]]; then exit "${code}"; fi
    return "${code}"
}
trap 'echo "Queue interrupted"; exit 130' INT
trap 'echo "Queue terminated"; exit 143' TERM
export WARMUP=2 MEASURE=4 VALIDATION_SAMPLES=1
export METHODS=native_n10,fixed_k2_r1,residual_k2,adaptive
echo 'Checking trace/checkpoint compatibility before waiting for GPU...'
OUT="${QUEUE_OUT}/00_smoke" bash scripts/profile_video_parallel.sh preflight > "${QUEUE_OUT}/preflight.log" 2>&1 || {
    cat "${QUEUE_OUT}/preflight.log"
    echo 'Preflight failed; no GPU job started.'
    exit 1
}
run_job 00_smoke profile || { echo 'Smoke failed; stopping before expensive tasks. See queue.log.'; exit 1; }
export WARMUP="${NIGHT_WARMUP:-10}" MEASURE="${NIGHT_MEASURE:-60}" VALIDATION_SAMPLES=3
export METHODS=native_n10,native_n4,native_n6,native_n8,legacy_n10,fixed_k2_r1,fixed_k4_r1,residual_k2,residual_k4,adaptive
run_job 01_main profile || true
export METHODS=native_n10,residual_k2,residual_k4,adaptive
run_job 02_tolerance sweep --tolerances 0.001,0.003,0.01 || true
export METHODS=native_n10,fixed_k2_r1,fixed_k2_r2,fixed_k4_r1,fixed_k4_r2,fixed_k4_r3,fixed_k4_r4
run_job 03_iterations evaluate || true
export METHODS=native_n10,residual_k2,residual_k4,adaptive
export SOLVER_OVERRIDES='{"local_guard":false}'
run_job 04_without_local_guard evaluate || true
export SOLVER_OVERRIDES='{}'
for seed in 43 44; do
    run_job "05_seed_${seed}" evaluate --seed "${seed}" || true
done
export DECODE_VIDEO=1 METHODS=native_n10,native_n6,residual_k2,adaptive
export MEASURE="${NIGHT_DECODE_MEASURE:-30}"
run_job 06_decode evaluate --video-samples 0 || true
echo "[$(date -Is)] Queue complete. Check status.tsv; nonzero jobs are failures, not results."
