# Action-aware video reuse: research implementation and experiment protocol

## Objective and current evidence

Test whether task-oriented video refresh can improve the closed-loop
success/latency frontier of an unchanged joint video/action policy. The candidate
uses the existing weights, bf16, native N10, original attention visibility and
normal action updates. It requires no training. Acceptance targets are at least
1.2× full-request speedup for feasibility, and preferably 1.5× with a paired
heldout success-rate lower confidence bound no worse than −2 percentage points.
These are research targets, not measured results or a publication guarantee.

The current E2 run is
`experiment/e2_3090_fresh_20261118_0137`. Its source identity remains untouched:
new runtime code is in `benchmarks/action_cache`, outside the original research
source scan. The new experiment separately binds its implementation hashes,
source model, observations and protocol. No existing sealed artifact is edited.

**Not yet established:** real-model cache validity, speedup, closed-loop quality,
cross-model generalization, or novelty relative to the latest literature. The
passing 3090 *weight-offload* preflight belongs to E2; it is not validation of
this new *activation-cache* method.

## What is implemented

- Request-local video K/V and residual-update cache for Wan mutual Joint.
  On reused blocks, future-video QKV, future-video attention query rows,
  cross-attention and MLP are skipped. First-frame token computations and the
  entire action branch are recomputed. Original K/V column order, absolute
  positions and sliced original attention mask are retained. The future video
  trajectory is approximate; its interaction with actions remains a test target.
- Policies: refresh-all, fixed interval, random refresh, feature-change refresh,
  sensitivity-only and sensitivity-plus-change. First and final steps refresh;
  maximum age forces refresh. Caches are cleared per request and on exceptions;
  nested/concurrent engine requests are rejected. CFG, asynchronous schedules,
  other caches, compile, VACE, prefix-KV and temporal patching are unsupported.
- Direct one-layer/one-step reuse probes, coarse layers 0/5/11/17/23/29 and
  steps 1/4/7. Effects are fitted only from all eight Discovery tasks with
  task-uniform means. No ranking is interpolated onto unmeasured layers.
  The adaptive score is a **testable heuristic**, not a calibrated risk bound:
  relative drift of fixed future-video hidden-state probes, multiplied by
  denoising-stage weight and measured sensitivity. Static and change-only
  controls separate the contributions.
- Fixed-observation replay with independently recomputed Native references,
  refresh-all validation, native EEF10 action/gripper error and future-video
  latent error. Complete-call synchronized timings include resets, input
  preparation, controller, cache bookkeeping and configured VAE decode.
  These are cold paired replay timings, **not** steady-state server timings.
- Separate native operator profiling: video Q/K/V/O, mixed attention, video
  text cross-attention, video MLP, action pre/post and condition preparation.
  Inclusive regions overlap; do not add them to operator times.
- Closed-loop screen/heldout job freezing over the existing pinned evaluator;
  independent policy seeds 43/44/45 preserve environment seed 42 and initial
  state manifests. Paired task/initial-state bootstrap keeps policy seeds
  together. Missing trials remain missing, and development runs cannot make
  a confirmatory noninferiority claim.

## Run the next stages

### Five-hour follow-up to the 3090 pilot

The completed cold pilot measured 0.98× paired request speedup for fixed-3
with 40% fewer video query rows. This follow-up measures prompt cache hits,
profiles fixed cache overhead, and tests whether real reuse error produces a
better group ranking than action attention.

Run these commands manually on one idle GPU. Keep the same shell variables
for all stages. Prepare a new protocol so it binds the new code. The phase
cutoffs below share one start time and leave the final 15 minutes for summary.

```bash
GPU_ID=0
RUN="experiment/ac_next_$(date +%Y%m%d_%H%M%S)"

bash scripts/research_action_cache.sh prepare \
  --e2-run experiment/e2_3090_fresh_20261118_0137 \
  --output "$RUN/protocol"

T0=$(date +%s)
bash scripts/research_action_cache.sh next --stage warm \
  --output "$RUN" --gpu "$GPU_ID" --memory-mode text-offload \
  --deadline "$(date -d "@$((T0 + 75*60))" --iso-8601=seconds)"

bash scripts/research_action_cache.sh next --stage overhead \
  --output "$RUN" --gpu "$GPU_ID" --memory-mode text-offload \
  --deadline "$(date -d "@$((T0 + 150*60))" --iso-8601=seconds)"

# Check group_gate in "$RUN/overhead/report.json".
bash scripts/research_action_cache.sh next --stage group \
  --output "$RUN" --gpu "$GPU_ID" --memory-mode text-offload \
  --deadline "$(date -d "@$((T0 + 285*60))" --iso-8601=seconds)"

bash scripts/research_action_cache.sh next-report --output "$RUN"
```

- **Warm.** Measure Native N10, Native N4, refresh-all, fixed-2 and fixed-3
  over 16 Discovery observations, seeds 43/44, and three paired repeats.
  Compare cold and hot N10 once per task. Text encoder hooks verify that
  measured hot requests hit the cache.
- **Overhead.** Compare the same fixed schedules with and without write
  elision, removal of unused feature probes, and shared mask slicing.
  The gate requires a complete paired panel and a fixed schedule with at
  least 1.10× joint forward and 1.05× full request speedup, both against N10.
- **Group.** At layer 29 and zero-based step 9, reuse K/V and block deltas
  from step 8. Fit the eight single-group scores on Discovery, then compare
  the resulting two-group selection with E2 G1/G5, step-8 attention and
  five random pairs on 16 Calibration observations from episodes outside
  the original E2 selection. Score the first eight actions to match closed
  loop E8 and measure gripper direction by sign.

Read `next_report.md` for the timing and group comparisons and
`next_report.json` for full statistics. Per-stage status and matched
records are stored under `warm/`, `overhead/` and `group/`.

### Original feasibility protocol

All GPU commands require an **idle, unlocked** physical GPU and an explicit
deadline. Existing experiments are never displaced. GPU IDs below are examples
to use only after an independent device becomes available. Primary performance
evaluation should use resident weights on an exclusive A100/H100; explicitly
select `--memory-mode text-offload` for a separate 3090 diagnostic panel.

### 1. Freeze feasibility protocol and audit E2 evidence

```bash
bash scripts/research_action_cache.sh prepare \
  --e2-run experiment/e2_3090_fresh_20261118_0137 \
  --output experiment/action_cache_protocol_20261118
```

`protocol.json` binds methods, code and the 8/8/24 task split;
`e2_evidence.json` records only observed progress. Recreate in a new directory
if the new implementation changes. Source E2 states can be replayed on a new
GPU, but model assets must match and native reference actions are recomputed;
this does not authorize resuming old E2 under a new identity.

### 2. Exclusive profiling and minimal real-cache pilot

```bash
bash scripts/research_action_cache.sh run \
  --protocol experiment/action_cache_protocol_20261118/protocol.json \
  --output experiment/action_cache_profile_NEW --gpu 0 --phase profile \
  --deadline 2026-11-19T10:00:00+08:00

bash scripts/research_action_cache.sh run \
  --protocol experiment/action_cache_protocol_20261118/protocol.json \
  --output experiment/action_cache_pilot_NEW --gpu 0 --phase pilot \
  --deadline 2026-11-19T10:00:00+08:00
```

The sample date is the current authorized cutoff; a later run needs a newly
chosen deadline. Add `--decode-video` for a separate VAE-on panel. Refresh-all
must match Native at unchanged `rtol=1e-4, atol=1e-5`; an unsuccessful check
stops rather than weakening thresholds. Native N4/N6/N8 are included as quality/
compute-budget baselines; only N10 is the speedup denominator.

The minimal queued 3090 pilot uses Native N10, refresh-all, fixed-2, fixed-3,
and change-0.1 over eight Discovery tasks with seed 43. It is labelled diagnostic
and cannot satisfy the full paper acceptance criteria.

### 3. Direct probes, fit, and Calibration comparison

```bash
bash scripts/research_action_cache.sh run \
  --protocol experiment/action_cache_protocol_20261118/protocol.json \
  --output experiment/action_cache_probes_NEW --gpu 0 --phase probe \
  --deadline 2026-11-19T10:00:00+08:00

bash scripts/research_action_cache.sh fit \
  --probe-run experiment/action_cache_probes_NEW \
  --output experiment/action_cache_sensitivity_NEW.json

bash scripts/research_action_cache.sh run \
  --protocol experiment/action_cache_protocol_20261118/protocol.json \
  --output experiment/action_cache_calibration_NEW --gpu 0 --subset calibration \
  --phase pilot --seeds 43 44 45 \
  --sensitivity experiment/action_cache_sensitivity_NEW.json \
  --deadline 2026-11-19T10:00:00+08:00
```

Compare all predeclared fixed/random/change baselines with static/adaptive
thresholds .03/.1/.3. Coarse probe scope is six layers, so static/adaptive methods
refresh all other layers. Expanding to all layers requires a new predeclared
probe protocol, not extrapolated scores. Report controller overhead and measured
kernel/call reductions alongside latency. Video query count is not a FLOP count.

### 4. Calibration closed-loop screen, then freeze heldout

After Calibration replay completes, explicitly select the candidate from its
quality/latency curve. For example, **only if supported by development evidence**:

```bash
.venv-openwam/bin/python -m benchmarks.action_cache.evaluation plan \
  --protocol experiment/action_cache_protocol_20261118/protocol.json \
  --pilot-run experiment/action_cache_calibration_NEW \
  --candidate adaptive_0.1 \
  --methods native_n4,native_n6,native_n8,fixed_2,random_2,change_0.1,static_0.1,adaptive_0.1 \
  --stage screen --output experiment/action_cache_screen_NEW --gpus 0
```

This writes a sealed manifest, without running it. Execute with:

```bash
.venv-openwam/bin/python -m benchmarks.action_cache.evaluation run \
  --manifest experiment/action_cache_screen_NEW/manifest.json \
  --deadline 2026-11-19T10:00:00+08:00
```

Screen: eight Calibration tasks × five initial states × three policy seeds per
method. A complete screen is required before a heldout manifest can be made.
Use the same `plan` arguments with `--stage heldout`, add
`--screen experiment/action_cache_screen_NEW/manifest.json`, and choose a new
output directory. Heldout: 24 tasks × 50 initial states × three seeds = **3,600
episodes per method**. Code, methods, weights, sensitivity files and task split
are frozen; changing them requires a new study, not retuning heldout results.

## Decisions and remaining research work

1. **Mechanism gate:** completed E2 robustness and highest-priority matched
   controls support a stable effect. Zero/tied or unstable rankings do not pass.
2. **Computational gate:** operator profile shows meaningful removable time;
   direct sequential cache reuse gives a useful error/budget curve and >=1.2×
   measured request speedup. If not, stop the fine-grained extension and diagnose
   kernel/controller/memory overhead before scaling evaluation.
3. **Closed-loop gate:** screen full success/latency curves, report regressions,
   and select a limited method set. Error thresholds alone do not qualify a
   method as safe. The heldout target is at most −2 pp success-rate degradation,
   supported by the paired interval, preferably with >=1.5× request speedup.
4. **Second platform/model:** acquire a matching RoboTwin checkpoint and a
   different supported joint architecture. The present adapter deliberately
   rejects unsupported backbones. LIBERO-plus is perturbation evidence, not a
   substitute for a second independent environment. Robot experiments and
   second-model adaptation are not executed by the current LIBERO manifest.
5. **Statistical design:** before heldout, estimate power using actual screen
   discordances and task variability; revise prospective sample size if needed
   and freeze again. Current 50 × 3 is a planned budget, not a power guarantee.
6. **Literature and novelty:** complete the verification matrix below before
   writing a novelty claim; no up-to-date web literature review has run here.

## Related-work verification matrix

For diffusion acceleration families such as DeepCache, PAB, TeaCache,
FirstBlockCache and FasterCache, verify original papers, current code and actual
compatibility before selecting implementations. Also cover diffusion-action
policy acceleration, joint video/action caching, and adaptive token/layer reuse.
For each record: exact citation/version, cached object, refresh rule, trained or
training-free, temporal scope, action-conditioned evaluation, supported model,
available code, adaptation required, and matched-budget tuning space. Add the
strongest compatible external baseline before claiming a method contribution.

Main risks to distinguish experimentally: sensitivity adds no value over feature
drift; tiny action errors compound in contact-rich closed loop; cached features
have timestep mismatch; prefix/action fresh computation reduces achievable
speedup; controller or cache memory removes the theoretical savings.

## Validation and artifacts

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-openwam/bin/python -m pytest -q scripts/inference_test/test_action_cache.py
```

Toy execution tests check native parity at full refresh, actual reduced video
query/projection/MLP rows, fresh actions/prefix, original mask row slicing,
maximum age, request isolation, exception restoration, direct probe targeting,
data-split leakage prevention and paired closed-loop statistics. They do not
replace the real GPU refresh-all/native gate.

New protocol and queue paths are recorded in `ACTION_CACHE_EXECUTION_20261118.md`.
