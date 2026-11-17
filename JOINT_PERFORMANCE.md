# Joint performance implementation status

## Endpoint-guided tail experiments

Independent `inference.joint_action_tail_init` and `inference.joint_video_tail_init`
accept `stage_residual` (unchanged default) or `endpoint_bridge`. Bridge changes
only newly appended tail candidates to `(1-sigma/sigma0)*previous_endpoint_tail
+ (sigma/sigma0)*coupled_noise`. Overlap, initial noise coupling, current clean
prefix, inactive constraints, and one batch=N forward per warm request remain
unchanged. No periodic refresh or extra correction is added. The alternative is
inspired by ParaFlow alignment, not a proven accuracy fix: endpoint bias can persist
and the default residual rule's finite-forgetting guarantee no longer applies.

```bash
bash scripts/ablate_alpha_joint_tail.sh plan
# Spatial task 0, five states x four arms = 20 episodes
GPU=6 bash scripts/ablate_alpha_joint_tail.sh diagnostic
# Existing output summary, no GPU work
COMPARE_ROOT=/absolute/path/to/output bash scripts/ablate_alpha_joint_tail.sh summary
```

Four arms: residual/residual, action bridge only, video bridge only, both bridges.
All use N10/H32/E8, phase video history, seed 42, cached schedule, shared conditions,
no compile/DiT cache/pixel decode. Configure `SUITES`, `TASK_IDS`, `CKPT_DIR`,
`CKPT_NAME`, `GPU`/`GPUS`, and `COMPARE_ROOT` as needed. Tail modes enter resume
signatures and sampler logs; do not mix different protocols in one directory.
Reports include per-trial success and steps, cold/warm request timing, videos,
and the source-labeled historical Native reference (not a current paired control).
At least 4/5 on task 0 is a screening target only; if no arm improves, keep defaults
unchanged and do not expand the campaign.

### Offline same-noise diagnosis

```bash
# Capture Native observations, then analyze previously failing trial 1
GPU=6 bash scripts/diagnose_alpha_joint_tail.sh
# Reuse captured observations; episode IDs are trace-relative
GPU=6 TRACE=/absolute/path/to/native_observations.npz EPISODE=1 \
ACTION_TAIL_INIT=endpoint_bridge VIDEO_TAIL_INIT=stage_residual \
bash scripts/diagnose_alpha_joint_tail.sh
```

Without `TRACE`, capture preserves all five spatial-task-0 trials to retain the
canonical environment RNG sequence, then selects episode 1. Capture requires one
server/session; retry sessions are rejected, never silently concatenated. `OUT`
selects a fresh output directory. `GPU` selects the capture and replay GPU; set
`DEVICE` to the logical replay index under CUDA visibility remapping.

Native and candidate replay the same observations. Warm candidate forwards also
construct an offline serial reference from exactly the same initial noise,
conditions, sigma grid, and constraints. References never commit to candidate
state and are never counted as deployment timing. Full per-stage errors distinguish
batch-partition differences from history-candidate differences and separate action
overlap/new E8 tail and video overlap/new latent. First-eight output errors are
recorded both in normalized coordinates and after checkpoint unnormalization plus
LIBERO EEF10-to-OSC7 conversion. Native-output comparisons use Native's own noise
policy and are labeled separately. Error magnitude alone does not establish the
cause of a closed-loop failure.

Read `diagnosis/requests.jsonl`, `diagnosis/diagnosis.md`, and `diagnosis/diagnosis.json`.
Reports record source/trace/checkpoint identity and explicitly invalidate timing
and policy-equivalence claims. rtol=atol=0.05 is unchanged. CPU tests verify that
references do not alter candidate results; CUDA/model verification is still required.

After quality screening, run the existing paired latency and strict partition/
shared-trajectory checks with the same tail settings:

```bash
GPU=6 TRACE=/absolute/path/to/observations.npz \
bash scripts/compare_joint_native_parallel.sh \
    --action-tail-init endpoint_bridge --video-tail-init stage_residual
```

Finite partition failure may explicitly continue with `ALLOW_PARITY_FAILURE=1`
as diagnostic timings only; shared-trajectory failure cannot be bypassed. A candidate
passing task 0 should next face a fresh matched Native comparison on task 3 before
any default change. No GPU experiment or large campaign is automatically started.

## Single-round quality recovery experiments

The experimental `inference.joint_history_scope=action_only` mode borrows
ParaFlow's action-history-only candidate construction: shift action stages by E8,
reconstruct video candidates from current-request noise and the current clean
prefix at each sigma, execute one full stage-batched Joint forward, then action
affine scan and video Euler prefix update. It **still predicts future video
latents under mutual attention**; it is not first-frame-only/isolated inference.
The interpolated future frames are candidate initialization, not observed future
frames or final predictions. The existing local RNG/noise-coupling policy is
retained to avoid mixing a noise-policy change into this ablation.

Action-only history requires one serial initialization per episode, then exactly
one batch=N forward per warm request. It stores no cross-request video trajectory.
The old `phase_video_action` default remains unchanged (two video-phase initializations).
No extra refinement rounds, periodic refresh, or fallback are added.

```bash
# Inspect only; does not load a model or write experiment outputs
bash scripts/ablate_alpha_joint_quality.sh plan
# Default small screen: spatial task 0, five states x four methods = 20 episodes
GPU=6 bash scripts/ablate_alpha_joint_quality.sh diagnostic
# Validate on another task using a fresh output directory
GPU=6 TASK_IDS=3 bash scripts/ablate_alpha_joint_quality.sh diagnostic
# Rebuild the report without GPU work
COMPARE_ROOT=/absolute/path/to/joint_quality_output bash scripts/ablate_alpha_joint_quality.sh summary
```

The four arms are old phase history with/without shared conditions and new
action-only history with/without shared conditions. All use N10/H32/E8, seed 42,
schedule caching, no compilation/DiT cache/pixel decode, and independent sequential
closed-loop evaluations. `SUITES`, `TASK_IDS`, `CKPT_DIR`, `CKPT_NAME`, `GPUS`/`GPU`,
and `COMPARE_ROOT` are configurable. Resume with the same output and protocol;
do not reuse an output from a different arm/configuration. No GPU exclusivity
guard is added to this closed-loop runner; select an available device yourself.

`comparison.md` and `comparison.json` include per-trial success and policy steps,
cold/warm request counts and timing, source-labeled historical Native results,
and a screening label. At least 4/5 on spatial task 0 is only a threshold for
further task-3 validation, not proof of restored benchmark success rate. Review
the existing per-arm videos for grasp/contact failures. Historical Native 5/5
is not silently treated as a fresh paired control, particularly on other tasks.

Run batch-partition and mandatory shared-condition trajectory checks and paired
speed measurements for the new history mode using the existing replay entry:

```bash
GPU=6 TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
    bash scripts/compare_joint_native_parallel.sh --history-scope action_only
```

The replay tests both N4 and N10. Existing strict tolerances remain unchanged;
finite partition failures may explicitly continue with `ALLOW_PARITY_FAILURE=1`
as diagnostic timings only. Sharing-trajectory failures cannot be bypassed.
No accuracy recovery, CUDA parity, or latency improvement is claimed before
running these actual experiments. All inference remains inside this repository.

## LIBERO N10 closed-loop success rate

Compare Native Joint N10 against the same optimized parallel N10 configuration
used in replay (schedule caching and shared condition projections enabled):

```bash
# Small check: 4 suites x task 0 x 5 trials x 2 methods = 40 episodes
GPU=6 bash scripts/compare_alpha_libero_n10.sh diagnostic
# Qualitative subset: spatial tasks 0 and 3, five initial states each, 20 episodes total
GPU=6 SUITES=spatial TASK_IDS=0,3 bash scripts/compare_alpha_libero_n10.sh diagnostic
# Full: 4 suites x 10 tasks x 50 trials x 2 methods = 4000 episodes
GPU=6 bash scripts/compare_alpha_libero_n10.sh full
# Resume using the same mode, or summarize without starting inference
COMPARE_ROOT=/absolute/path/to/compare_n10_output bash scripts/compare_alpha_libero_n10.sh summary
```

Both methods use seed 42, H32/E8, synchronous execution, the same checkpoint,
future video latent denoising, no pixel decoding, no compilation, and no DiT cache.
They run sequentially and execute generated actions in LIBERO, unlike fixed-trace
replay. This is an evaluation of an experimental approximate policy, not a claim
that the failed batch-partition check was resolved. No offline parity gate is
required to launch this quality measurement. It does not bypass numeric safety
checks within the sampler. Diagnostic success rates are small-sample checks only.

Outputs default to `outputs/libero/compare_n10_<timestamp>` with per-method
results and `comparison.json`/`comparison.md`. `COMPARE_ROOT`, `CKPT_DIR`,
`CKPT_NAME`, `GPUS` (or `GPU`), `REPLICAS_PER_GPU`, and `BASE_PORT` can be overridden.
Set `COMPARE_ROOT` explicitly to resume interrupted runs with the same protocol;
do not mix full and diagnostic outputs. Optimization overrides are recorded in
the manifest and included in resume compatibility checks.

## Native Joint N10 versus current parallel Joint

Run only the current repository's Joint implementation (no ParaFlow checkout/runtime):

```bash
GPU=7 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/compare_joint_native_parallel.sh
```

This compares Native serial N10, parallel N10, and parallel N4. Both parallel
variants enable schedule caching and request-scoped shared condition projections.
N10 fixes the stage count, **not algorithm/output equivalence**; N4 additionally
reduces the stage count. Every variant denoises future video latents under the same
checkpoint's mutual attention, H32/E8, and input trace; none decodes video pixels.

Defaults: 48 warmup requests, 120 eligible measured requests, exclusive selected GPU,
no profiler. Foreign GPU processes cause failure and are never terminated. Without
`TRACE`, the existing LIBERO capture runs automatically before replay. `GPU` is the
physical selection; with CUDA visibility remapping, also set logical `DEVICE`.
Override `OUT`, `CKPT_DIR`, `CKPT_NAME`, `WARMUP`, or `MEASURE` as needed. Each output
directory must be fresh. Output defaults to `outputs/joint_native_parallel_<timestamp>`.

Read `comparison.md` for P50/P95 latency, CUDA intervals, denoising forward span,
peak allocated memory, and ratio-of-P50 speedups; `timing.json` additionally contains
paired request speedups and sampler timings where available. The forward span runs
from first joint forward start to last forward end, includes intervening gaps, and
excludes the last solver update; it is not a complete solver-only timing.
`requests.jsonl` flags unshared validation trajectories with `reference_only`.
Those run outside each reported method's timing interval, but remain memory-resident.

N4 and N10 each require stage-batch partition checks and mandatory full-trajectory
checks against unshared parallel execution. A finite partition mismatch may continue
only with explicit `ALLOW_PARITY_FAILURE=1`; results are then diagnostic timings,
not numerical-equivalence evidence. Shared-condition trajectory failures or nonfinite
outputs always stop. No closed-loop success rate is measured by this replay.

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

- `joint_shared_n4` adds stage-shared text and cross-attention condition projections,
  inspired by original ParaFlow's shared conditioning path, implemented entirely locally.
  The sampler option is `share_conditions=True`, disabled in deployment defaults.
  Text batch equality is checked before sharing; downstream K/V and K normalization
  share only stride-zero inputs. Queries, self-attention, future video tokens, time
  modulation and action generation are unchanged. Hooks are scoped to a single sampler
  invocation, restored on failure, and never retain conditions across observations.
  This stateful sampler assumes serialized requests on its architecture.
  The existing one-command profiler automatically includes this sixth method and
  enforces full-trajectory parity against unoptimized Joint at every observation.
  Real BF16 GPU parity, latency gains and closed-loop quality remain unvalidated.
  The separate short profiler now records `joint_shared_n4` (schedule caching off),
  with its method/configuration explicitly saved in `profile.json`; it is not a
  paired profiler comparison against historical operator totals.
- Optional schedule cache (`--cache-schedule` in replay), disabled in deployment defaults.
  It avoids repeated schedule allocation, grid validation and device-to-host scan construction.
  Cache misses compute normalized sigmas on the same device as the reference path, then retain
  those tensors. They do not substitute CPU division for GPU division. This preserves reference
  rounding; the first cache fill still incurs the reference validation/copy overhead.
- Transactional sampler failure: histories, request counter and both RNG streams roll back.
- Replay records stage batch sizes, peak allocated/reserved memory, free CUDA memory,
  torch/device metadata, input hash and implementation identity.
- Profiler exports CPU/CUDA operators and input shapes; `joint.forward` labels joint traversals.
  Profiling is deliberately separate from steady-state timing.
- CPU tests compare 64 requests, resets, changed prompts/prefixes, and failed-request retry.

No GPU speedup is claimed: the execution sandbox cannot initialize host CUDA.
Cached trajectory failures additionally save `cache_parity_failure.json` with action/video errors
and a bitwise normalized-sigma comparison. These failures remain fatal even in diagnostic mode.
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

Request-local prepared-conditioning caches and mutual prefix prefill are now implemented
as opt-in `inference.joint_execution_backend=prepared|prefix` paths. The default remains
`legacy`; these are not implicitly enabled by the schedule cache. Real-checkpoint CUDA
validation and performance remain not_run in the execution sandbox. CUDA graph/compile
and replicated two-GPU stage splitting are still **not implemented** by these paths.
The schedule cache does not eliminate finite checks or per-request synchronization.
Two-GPU experiments should follow profiling and numerical validation, with explicit device
assignment and rank-zero trajectory/RNG ownership, rather than per-layer model parallelism.

## LIBERO isolated-mask comparison

`scripts/compare_alpha_libero_isolated.sh` compares three closed-loop methods using the same
current LIBERO checkpoint: `mutual_native_n10`, `isolated_native_n10`, and
`isolated_paraflow_n10`. The isolated methods rebuild the attention driver after loading
the unchanged weights, using isolated attention and first-frame-causal video attention.
This is a **mask-switch ablation of mutual-trained weights**, not reproduction of the
original isolated-trained RoboTwin checkpoint or its success rate.

```bash
bash scripts/compare_alpha_libero_isolated.sh plan
bash scripts/compare_alpha_libero_isolated.sh preflight
GPU=6 bash scripts/compare_alpha_libero_isolated.sh diagnostic
```

The default diagnostic runs spatial task 0, five matched initial states per method:
15 episodes total, seed 42, N10, H32, E8, synchronous execution, no compile, DiT cache,
or video decoding. `SUITES` and `TASK_IDS` select other task subsets. Results are written
to `outputs/libero/isolated_ablation_<timestamp>/comparison.md` and `comparison.json`.
To regenerate a report, set `COMPARE_ROOT` to that directory and use `summary`.
Five episodes are qualitative diagnostics, not a benchmark-level accuracy estimate.

The isolated methods load original ParaFlow operators from
`/home/zhijian/paraflow-main` and native stage/history helpers from
`/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port/native/openwam`.
Override these with `PF_ROOT` and `PF_NATIVE_ROOT`. Shared sources remain read-only;
source hashes, checkpoint identity and inference overrides participate in resume identity.
The current repository remains the active `openwam` package.

Both isolated methods prefill the current frame and generate actions without future-video
denoising. Mutual Native generates future-video latents as well. Interpret mutual versus
isolated as a mask/quality ablation, **not** an equivalent-workload speedup.
Within isolated, Native serial versus original ParaFlow measures the algorithm change;
their original per-request versus shifted-history noise policies are retained.

For paired timing on captured E8 observations, run separately:

```bash
GPU=6 TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
  bash scripts/compare_alpha_libero_isolated.sh replay
```

Replay requires an exclusive GPU, uses independent sampler histories on matched observations,
and writes its own report under the comparison directory's `replay` subdirectory.
First-frame versus full-isolated frozen-candidate action velocities and warm stage-batch
versus independent-stage velocities are gated at rtol/atol 0.05. Failures stop by default.
`ALLOW_PARITY_FAILURE=1` permits finite diagnostic results only; it does not establish
equivalence. First-frame execution has no future-video block calls; this call budget and
validation diagnostics are recorded in server logs. Import preflight does not validate
GPU kernels or model parity. CPU tests pass, but real-model CUDA validation and success
rates must be established by running the commands on the host; no GPU run is launched
automatically.

## Mutual execution ablation and paired replay

The current Joint solver/history strategy is unchanged. Prepared execution computes
text/proprio conditioning and each branch's cross-attention K/V once per observation.
Prefix execution additionally prefills the independent clean frame once, then updates
only future-video/action query rows. Full ordered K/V, mutual attention, absolute RoPE,
token-local time modulation and the complete video output head are retained. Prefix
cache lifetime is one request. Unsupported conditioning is rejected, not approximated.

At N10 with 360 video tokens (120 clean prefix) and 32 action tokens per stage,
the block token-instance budget changes from 3920 to 2840, including 120 once-only
prefix instances. This is not a measured 27.6% latency reduction. The warm solver
still has one batched velocity traversal; prefill is a separate observation-only pass.
Cold initialization remains serial N10. Strict frozen velocity checks and separate
stage-batch partition checks add diagnostic forwards on first use and episode resets;
those calls are excluded from formal replay timing, not silently counted as model work.

```bash
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh execution-ablation
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh execution-regression
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh execution-all
```

The first command runs five methods x three tasks x five paired scenes (75 episodes).
The second runs legacy/prefix on three tasks x twenty scenes (120 episodes).
The third combines short ablation, independent real RoboTwin request capture and
validated fixed-observation replay/profiling. It does not run the 120-episode regression.
See `benchmarks/robotwin/README.md` for `capture`/`replay`, provenance validation,
strict parity failure behavior and output paths. Replay uses independent histories
to validate complete trajectories and executed actions, then matched legacy-owned
candidates/RNG to time execution alone. Profiler/event attribution runs separately.
Acceptance requires at least 10% P50 reduction without P95 regression, passed numeric
gates and no same-run legacy success-count regression. No result changes the default
backend automatically, and no GPU improvement is claimed before host measurements.
