# Joint performance implementation status

## Reproducible measurement

The observation input is a real E8 request trace, not a synthetic performance workload.
Use an NPZ containing composed RGB `images` (uint8 N,H,W,3), string `prompts`,
finite `states` (N,D), and boolean `episode_starts` (N, first true).
Each row must correspond to an actual action chunk request, not every simulator step.
The unified runner now records these inputs automatically at `engine.generate`, so buffered
action execution does not duplicate observations. It saves only successful generation requests,
preserves episode/prompt boundaries, and aggregates short server sessions without pretending
they are one continuous rollout.

```bash
bash scripts/profile_alpha_joint.sh plan
# Complete LIBERO capture + numerical gates + paired timing + separate profiler + report:
GPU=0 PROFILE_ROOT=outputs/joint_experiment_libero bash scripts/profile_alpha_joint.sh all
# Complete RoboTwin equivalent, using the locally configured RoboTwin installation:
BENCHMARK=robotwin GPU=0 PROFILE_ROOT=outputs/joint_experiment_robotwin \
    bash scripts/profile_alpha_joint.sh all
# Reuse a real trace, skipping simulator work:
TRACE=/absolute/path/to/observations.npz GPU=0 \
    bash scripts/profile_alpha_joint.sh all
```

`BENCHMARK=robotwin` defaults to `/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full`;
LIBERO defaults to `/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO`. Checkpoint discovery requires
exactly one checkpoint safetensors file; otherwise set `CKPT_NAME`. Built-in RoboTwin capture
uses its existing diagnostic launcher, which serves the latest weight; an explicitly selected
older weight requires a matching external trace. RoboTwin may first build its diagnostic manifests.

The default mode is `all` and **starts GPU work**. Use `plan` to inspect without running.
`GPU=0` selects policy GPU 0 and defaults replay to `cuda:0`; `GPU=3` defaults replay to `cuda:3`.
With an existing `CUDA_VISIBLE_DEVICES` remapping, set `DEVICE` explicitly to the corresponding
logical device. `PORT` defaults to 8950. Select an available GPU yourself: the runner does not
reserve the device or kill unrelated jobs. Only this run's owned collection/worker subprocesses
are stopped. GPU/process snapshots are saved, not treated as proof of exclusivity.

Automatic LIBERO capture uses Native N4/E8 on spatial task 0 with up to 50 trials; select
`SUITE`/`TASK_ID` to change that observation workload. RoboTwin uses the existing three-task
diagnostic collector. Capture stops once enough eligible requests are available. Its partial
evaluation outputs are **not success-rate measurements**. `CAPTURE_TIMEOUT` defaults to 7200 seconds;
collection failure/insufficient observations stops the experiment rather than repeating frames.

Each run uses a fresh `PROFILE_ROOT`; results are never silently overwritten or resumed.
Run stages separately with `capture`, `timing`, `profile`, or regenerate a completed timing report
with `PROFILE_ROOT=/existing/run bash scripts/profile_alpha_joint.sh summary`.
`timing` and `profile` require `TRACE`; reuse the original run's `observations.npz` in a fresh directory.
`SKIP_PROFILE=1` disables the separate profiler in `all` mode.
For a short **pipeline check**, use `WARMUP=2 MEASURE=3 MICRO_WARMUP=1 MICRO_MEASURE=2`;
such a small sample is not a publishable latency benchmark. The default full measurement
is the recommended comparison workload. This runner does not launch a full success-rate campaign.

Timing measures composed canvas/proprio to engine result, excluding simulator and camera composition.
Native and Joint both exclude warmup and the first two requests following episode/prompt changes,
using the same phase-bootstrap exclusion window. Joint additionally checks its actual cold flag.
Insufficient traces fail preflight. The older standalone `replay_joint.py` still marks them
`insufficient_warm_observations`, but is not the new unified runner.
Five methods now share one loaded model, with independently retained Joint histories, noise
generators, condition caches, and CPU/CUDA RNG state. Method order is deterministically shuffled
for every observation; all methods use exactly the same measured indices. Global warmup is 48
requests, followed by 120 eligible observations by default (`WARMUP`/`MEASURE`).
Before timing, batch4 is checked against 4×batch1 and 2×batch2, then all three partitions are
microbenchmarked on identical frozen stage inputs (5 warmup, 20 measurements by default;
`MICRO_WARMUP`/`MICRO_MEASURE`). This is forward-only timing, not Native trajectory integration.
Cached-vs-reference Joint video/action trajectories are checked after every observation outside
the timing boundary. The thresholds stay at rtol=atol=0.05. Stage-batch failures save both modalities'
mismatch counts/fractions, max/mean absolute error, RMSE and relative L2 to `parity.json` before
the default strict abort. BF16 batch-dependent kernels are one possible source of differences,
not an established explanation for any particular mismatch.

For performance investigation despite **finite** stage-batch mismatches, explicitly set
`ALLOW_PARITY_FAILURE=1`. This does not change thresholds or mark parity as passed:
`timing.json` records `numerical_equivalence_validated=false` and
`interpretation=performance_diagnostic_only_parity_failed`; Markdown displays the failure.
Nonfinite predictions and cached-trajectory mismatches remain fatal. Reuse a completed capture:

```bash
GPU=0 ALLOW_PARITY_FAILURE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/profile_alpha_joint.sh all
```

Such a run supplies bottleneck measurements, **not** an equivalence validation. Strict mode remains
the default. Investigate numerical drift separately before deploying a new execution path.
The profiler runs in a **separate process after timing**, avoiding profiler-contaminated comparisons.
GPU thermal/load drift can still occur; interleaving reduces, but does not eliminate, that confounder.
Native N1 is a latency reference, not evidence of equal policy quality.

## Output files

- `comparison.md`: timing/memory comparison and frozen-stage microbenchmarks.
- `timing.json`: paired per-observation speedups/differences, cold starts, hashes and device metadata.
- `requests.jsonl`: measured/warmup rows, method order, input/token shapes, allocated/reserved/free memory.
- `parity.json`: batch partition numerical checks and repeated forward microbenchmarks.
- `joint_trace.json`, `operators.txt`, `profile.json`: separate profiler trace, operator/shape ranking,
  and observed attention implementation events, with preparation/VAE/conditioning/FFN scopes.
- `observations.npz`, `capture.json`, `capture_shards/`, `capture.log`, `capture_eval/`: automatic
  collection data, source sessions and diagnostic logs (absent when reusing an external trace).
- `experiment.json`: explicit running/completed/failed status and error message.

PyTorch peak memory includes one resident model plus all methods' independent small histories
and conditioning caches. Reserved memory is process-wide/high-water-mark affected, not a fresh
per-method allocation requirement. CUDA event elapsed time can include idle gaps, so use the
operator trace rather than treating the event number as pure kernel time.

## Implemented adapter changes

- Optional schedule cache (`--cache-schedule` in replay), disabled in deployment defaults.
  It avoids repeated schedule allocation, grid validation and device-to-host scan construction.
- Transactional sampler failure: histories, request counter and both RNG streams roll back.
- Replay records stage batch sizes, peak allocated/reserved memory, free CUDA memory,
  torch/device metadata, input hash and implementation identity.
- Profiler exports CPU/CUDA operators and input shapes; `joint.forward` labels joint traversals.
  Profiling is deliberately separate from steady-state timing.
- CPU tests compare 64 requests, resets, changed prompts/prefixes, and failed-request retry.

No GPU speedup is claimed: the execution sandbox cannot initialize host CUDA.
Existing running evaluators are not modified or restarted; new source hashes may invalidate
resume against old manifests. Keep old experiment outputs separate.

## Bottleneck interpretation and remaining work

The referenced 447.077 to 110.624 ms result in
`/home/zhijian/paraflow-main/results/20261017-hardware-ablation/evidence/openwam-formal-host2-v6r/summary.json`
uses N10 and a first-frame-only isolated workflow. It is not the same computation as full mutual
Joint future-video denoising. It is not a one-forward Native baseline.

The current full Joint configuration has 360 video tokens plus 32 action tokens per stage.
N4 batching is four independent 392-token sequences, not one 1568-token sequence.
LIBERO and RoboTwin use the same composed canvas and model dimensions; additional active
action coordinates do not double the model token count. Approximately 23.6 GiB peak PyTorch
allocation alone does not establish memory pressure on an 80 GiB A100; reserved/free memory
and actual kernels must be examined before selecting multi-GPU execution.

The next architectural optimization is observation-local conditioning: encode proprio/context
once, reuse per-layer text K/V across stages, and compute TI2V time modulation at unique times.
Then prefill the 120 clean-prefix tokens once per observation, retaining their per-layer K/V.
Full future video/action queries must still attend the same keys, masks and FP64 RoPE positions.
This reduces video linear token instances from 1440 to 1080 at N4, not necessarily total latency
by 25%. It requires real-model parity tests before deployment.

Prepared conditioning, prefix prefill, CUDA graph/compile and replicated two-GPU stage splitting
are **not implemented in this patch**. They must not be represented as enabled by the schedule
cache. The latter also does not eliminate all finite checks or per-request synchronization.
Two-GPU experiments should follow profiling and numerical validation, with explicit device
assignment and rank-zero trajectory/RNG ownership, rather than per-layer model parallelism.
