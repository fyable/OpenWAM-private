# Original ParaFlow / Joint optimization benchmark

## Historical 110 ms controlled baseline

This is separate from the current API-mode sweep below:

```bash
bash scripts/reproduce_paraflow_110.sh prepare
GPU=5 bash scripts/reproduce_paraflow_110.sh all
```

The runner extracts the archived 20261017 `wam` source into a fresh local output
directory and verifies recorded source hashes. It does not edit the ParaFlow repo
or shared native runtime. The original controlled stage-0 EagerForward engine runs
with graph/cache/fusion/tuning disabled; only the baseline and original Native N10
reference are selected, not C1-C5. Numerical checks, exclusive GPU acquisition,
48 warmup / 120 samples / 64-request validation stream are retained.
Historical overlay records are copied provenance, not evidence of a new validation.
Frozen code/runtime compatibility and actual GPU timings remain to be validated;
110.624 ms is a reference, never a promised rerun latency. Inspect `run.log`,
`measurement/summary.json`, and `reproduction.json` in the new output directory.
The historical allocator waits up to one hour for the selected GPU to become available.

## Run

### Add future-video generation to the historical workload

```bash
GPU=7 bash scripts/profile_paraflow_with_video.sh
```

This separately measures the historical controlled first-frame baseline, original
full-video sampler with action-only history, and original phase-aligned joint history.
All use Isolated9890/N10/H32/E8 with unchanged isolated attention masks. Future video
is generated but cannot influence actions through these masks: this is not mutual.
The full-video path cannot use the isolated-only prefill engine; it runs native
operators. Differences therefore include execution-path changes, not a pure additive
video cost. No claim of exact algorithmic equivalence is made between the variants.
Graph/fusion/tuning remain off. Frame decoding remains off: this measures future
video **latents**, not RGB video delivery. Each variant requires repeated-reset
action repeatability and finite outputs; every generated video output is checked
for finiteness, and full multi-frame forward calls must be observed. These checks
do not validate video quality or equivalence to serial generation.
The launcher changes only the diagnostic orchestration via a recorded local AST
instrumentation; frozen model/sampler implementations are not edited. Results are
in `measurement/video_summary.json` and `run.log`. The setup still creates the
historical shared-weight model views; each variant is measured sequentially.

```bash
cd /home/zhijian/OpenWAM-main
bash scripts/compare_joint_original_paraflow.sh plan
GPU=7 TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
    bash scripts/compare_joint_original_paraflow.sh preflight
GPU=7 ALLOW_PARITY_FAILURE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
    bash scripts/compare_joint_original_paraflow.sh all
```

`all` starts explicit GPU experiments. Choose an idle device; both panels reject foreign
compute processes on the measured device. Nothing kills unrelated jobs. CPU/model/kernel
dependencies may still fail after import preflight; an import check is not GPU validation.

Defaults reuse this server's read-only original runtime and Isolated9890 assets under:

```
/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port/
  runtime/wam/bin/python
  native/openwam/
  assets/openwam/
  original-options/openwam-observation.npz
```

ParaFlow code comes directly from `/home/zhijian/paraflow-main/src`; it is not reimplemented
in this runner. The working directory and import assertions prevent this Joint checkout
from shadowing original OpenWAM. Override `PF_ROOT`, `PF_PYTHON`, `PF_NATIVE_ROOT`, or
`PF_OPTIONS` explicitly when moving machines. The options JSON uses the official factory
schema: absolute `checkpoint`, absolute `observation`, optional `checkpoint_name`, `operators`.

## Protocol

- Calls original `paraflow.infer.benchmark` with `paraflow.infer.openwam.replay:create_case`.
- N4 and N10, H32/E8, 48 warmup, 120 measured requests, correctness validation and GPU
  exclusivity required. It uses original raw-camera replay and input changes as provided
  by that factory; no fabricated numerical receipts.
- Sweeps supported isolated OpenWAM modes: cuda_graph, model_fused, model_deep, model_deep_tuned.
  The original isolated backend does not support eager (its error misleadingly mentions device count).
  Non-tuned modes use portable operators with refine_graph/gpu_pixels/early_vae disabled;
  only model_deep_tuned enables refine graphs and GPU preprocessing by default.
  Its default plan is now **portable**: the historical accepted plan requires an exact
  Torch/Triton/GPU/cuBLASLt match, and this server reported a cuBLASLt mismatch.
  This is an explicit default configuration, not a silent retry or a reproduction of
  accepted fixed kernels. To opt into accepted, supply `PF_OPTIONS` with
  `operators.plan="accepted"` after restoring the matching runtime libraries;
  mismatches remain fatal. Never bypass the original version guard.
  These are **not** claimed to recreate historical cumulative C1-C5 toggles exactly.
  `PF_MODES=cuda_graph,model_deep_tuned` or `PF_STEPS=10` narrows the original sweep.
- Current Joint then runs its existing native N1/N4/N10 and parallel N4/cache matrix,
  using an independent process on the same visible GPU, followed by a separate profiler.
- `ALLOW_PARITY_FAILURE=1` is optional and affects only Joint's finite stage-partition
  diagnostic gate. A failed gate remains visibly failed; cached trajectory parity is fatal.
- Results use a fresh `COMPARE_ROOT` (default timestamped outputs directory). Original
  per-mode logs, options and official receipts are retained. The top-level `comparison.md`
  separates panels and computes cuda_graph-relative ratios only within ParaFlow at the same N.
  This accelerated reference is neither eager nor a reproduction of the historical 110 ms baseline.
  The report includes the actual per-mode operator plan. GPU execution of portable
  tuned mode must still pass the official numerical validation before timings count.

## Fairness boundary

This benchmark fairly reports **within-workload optimization effects**, not a same-model
head-to-head algorithm comparison. Original ParaFlow uses its isolated checkpoint and
first-frame-only semantics, and raw-camera-to-action boundary. Current Joint uses a mutual
checkpoint, future-video generation and composed-canvas replay. They also consume different
observations and use different warm/cold selection protocols. Do not subtract their module
times or interpret a cross-panel ratio as an algorithm speedup. Original modes execute in
separate processes, not paired/interleaved, so repeat independent runs for robust estimates.

Joint now includes `joint_shared_n4`: schedule caching plus request-scoped sharing of
identical stage text projections and cross-attention K/V projections/normalization.
It checks contiguous text inputs for exact batch equality before sharing, and shares
downstream projections only for stride-zero batches. Different stage conditions fall
back to native execution. Its complete action/video histories must pass the original
0.05 atol/rtol trajectory gate; `ALLOW_PARITY_FAILURE` cannot bypass this new gate.
There is no cross-request K/V cache, prefix prefill, compilation or CUDA Graph for Joint.
Neither panel measures closed-loop quality; historical latency reproduction is not asserted.
A strict same-workload comparison still requires a validated original mutual backend or an
independently validated first-frame port; this runner does not pretend either exists.
