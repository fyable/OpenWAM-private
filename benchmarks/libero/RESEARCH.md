# OpenWAM-α H1/H2 research framework

This is an opt-in **diagnostic framework**, not a faster inference backend.
It does not establish H1/H2, closed-loop non-inferiority, or any speedup.
Existing native, joint-paraflow, and official DiT-cache implementations remain
available. Research mode rejects cache, compile, asynchronous execution,
non-native sampling, CFG other than 1, unsupported architectures and ambiguous
prefix/token layouts. No model downloads, environment changes, or checkpoint
modifications are performed automatically.

## Implementation and interfaces

The Python interface is `engine.generate({**conditions, "research_context": context})`,
where `context` is a fresh `ResearchContext(Intervention(...))`. A plain intervention
dictionary is also accepted by the engine; retain a context object when accessing
diagnostics in Python. `configs/research/intervention.example.json` illustrates
the dictionary, **not** a deployment YAML or server command-line flag.

Each generation has its own context and immutable step descriptors. Nothing is
installed on the driver between requests. The context is single-use and stores
only selected-layer summaries, action predictions, layouts and final normalized
actions. Baseline feature collection additionally captures selected V tensors on
CPU, one layer/request at a time; reduce the selected layers/states if host memory
is insufficient. This is intentionally outside deployment timing.

Attention runs the original full SDPA call, retaining its original outputs for
all unselected rows. Selected contributions are computed from FP32 probabilities
with the original visibility mask and **original denominator**, then added as a
correction. Only selected query rows are materialized, in chunks (default 32);
no full video-by-video attention matrix is stored. Reverse controls select only
future-video query rows reading actions. Gate 1 is an identity correction.
There is no token removal, kernel acceleration, or FLOP-saving claim.

`value_noise` changes only selected video V contributions to action queries:
Q/K and video outputs are unchanged. `epsilon`, `floor`, `sign`, and `noise_seed`
control an equal-relative-energy perturbation using a private generator.
`value_stale` instead replaces those values with the preceding baseline step's
matching values, records its **measured** relative energy and source sigmas, and
requires `sign=1`. This is not sigma-matched evidence: analyze its noise-level
mismatch separately. Different group sizes are avoided by divisor-based
temporal/spatial partitions; fewer than eight groups are possible. Grouping is
latent time × spatial row bands, **not inferred camera identity**.

The ordinary `seed` still controls both modalities. Independent video/action
random sources and E3/E4 are deferred. So are deployable rule learning, T3/T4
closed-loop evaluation, confirmatory SR statistics, and performance benchmarks.

## 1. Preflight and freeze the task split

### Local one-command launcher

For an overnight independent-task replication after Discovery E1, use
`E1_RUN=/absolute/discovery/run GPU=0 bash scripts/research_alpha_libero.sh calibration`.
It copies the frozen split, collects ten episodes per Calibration task, runs
its own T0 and smoke, screens all 24 E1 interventions plus five baselines, and
writes `tables/e1_calibration`. It never opens held-out tasks, automatically
selects E2 positions or claims closed-loop quality. The original Discovery
artifacts remain unchanged. Use `nohup` and a fresh `RUN` to survive disconnects;
this pipeline stops on errors and has no implicit partial-run resume.

On this server, after installing the LIBERO simulation dependencies into
`.venv-openwam`, run `bash scripts/research_alpha_libero.sh check` to verify an
actual offscreen environment reset/step and the CUDA/model imports. Run
`GPU=0 bash scripts/research_alpha_libero.sh e1` to collect Discovery states,
run T0, generate the E1 plan, run one intervention smoke, screen and summarize.
The launcher defaults to the existing checkpoint directory
`/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO`, checkpoint
`checkpoint_step_10690.safetensors`, and a fresh `experiment/h1h2_*` directory.
Override `RESEARCH_PYTHON`, `CKPT_DIR`, `CKPT_NAME`, `GPU` or `RUN` explicitly
when needed. `launcher.log` records progress; errors stop the pipeline.
No simulator or checkpoint download is performed. This is a diagnostic run,
not a deployment benchmark or a closed-loop non-inferiority test.

Simulation additions are listed in `benchmarks/libero/requirements-research.txt`;
they do not replace the model environment's CUDA PyTorch. The local robosuite
`macros_private.py` disables its shared `/tmp/robosuite.log` file sink to avoid
permission conflicts with other users. LIBERO configuration, matplotlib, numba
and shader caches are routed under the workspace rather than a read-only home.

E2 still requires an E1-informed frozen selection (section 4). With that artifact,
`E1_RUN=/absolute/e1/run SELECTION=/absolute/selection.json CALIBRATION_BANK=/absolute/calibration/bank CALIBRATION_T0=/absolute/t0_certificate.json bash scripts/research_alpha_libero.sh e2`
fits feature scale and runs E2. First collect a Calibration bank and pass its T0
using the CLI below. Selection state IDs must refer to the actual banks; do not
invent Calibration IDs before collection. The E2 mode is not an automatic
substitute for scientific position/state selection.

Run from the repository root using the model environment:

```bash
python -m benchmarks.libero.research --help
python -m benchmarks.libero.research preflight \
  --ckpt-dir assets/openwam_ckpt/openwam_alpha \
  --output experiment/manifests/preflight.json
```

All output paths are exclusive: reruns must use a new directory/file, not
overwrite an earlier run. Preflight does not load a model. It hashes local
Python/config sources and checkpoint-directory assets and records missing
dependencies/CUDA. Model loading during later commands additionally records
the selected weight hash, resolved config/hash, actual mask, active mapping,
binary-command dimensions and execution horizon. The reference upstream commit
is recorded but **local equivalence is not claimed**, even without a Git HEAD.

Model identity preserves the complete merged config as `config_unresolved`.
`resolved_config` resolves all runtime sections, excluding the training-only
`project` and `training` sections, which remain in the raw record and identity
hash. This avoids evaluating training output-directory `${now:...}` expressions
or introducing wall-clock-dependent hashes across collection and replay.
Runtime interpolation errors still stop the run rather than being ignored.

Export `tasks.json` from the installed LIBERO benchmark in its actual task-ID
order. Set up LIBERO as in the existing benchmark README first:

```bash
python - <<'PY'
from libero.libero import benchmark
from openwam.research.protocol import SUITES, write_json
tasks = {}
for name in SUITES:
    suite = benchmark.get_benchmark_dict()[name]()
    tasks[name] = [suite.get_task(index).name for index in range(10)]
write_json("experiment/manifests/tasks.json", tasks)
PY
python -m benchmarks.libero.research split \
  --task-map experiment/manifests/tasks.json \
  --output experiment/manifests/split.json
```

The split is PCG64 seed 20261112, fixed suite order, and 2/2/6 tasks per suite.
All analysis/collection commands reject held-out tasks. Seals are SHA-256
integrity checks, not proof that a manually declared selection is leakage-free.

## 2. Collect development states

Supply an explicit official LIBERO task checkpoint name, not foundation weights.
The examples retain the configured execution horizon; if using `--horizon`, use
the **same value for every baseline, collection, T0 and replay command**.
The loader's resolved configuration remains authoritative.

```bash
python -m benchmarks.libero.research collect \
  --ckpt-dir CHECKPOINT_DIRECTORY --ckpt-name CHECKPOINT.safetensors \
  --split experiment/manifests/split.json --selected-split discovery \
  --seeds 42 --episodes 10 --output experiment/state_bank/discovery
```

Repeat with `--selected-split calibration` and a different output directory.
The collector uses the existing LIBERO policy adapter and a local transport to
the unchanged server preprocessing/policy/executor, not a new camera or action
conversion implementation. It uses the existing policy configuration (600/700
task limits and settling), the first distinct initial-state IDs, and retains
failed episodes. It refuses custom clipping/index projection.

Recording happens only at actual chunk-generation boundaries; the action queue
is empty at those boundaries. At most three distinct states are selected near
20%/50%/80% of each episode's actual request sequence. Short episodes are not
duplicated. `phase=progress_proxy` is **not a contact annotation**. Offline labels
may be added in a new, resealed bank manifest, never fed to the model; keep the
original bank for audit and rerun T0 if its hash changes.

Each selected NPZ contains the composed image, raw proprio, prompt, baseline
chunk and available simulator state. Sidecars record policy/global RNGs,
initial-state identity, task progress and unverified controller/environment
state limitations. `closed_loop_restore_verified=false` is deliberate: these
states support open-loop replay, **not certified mid-episode simulator restore**.
Future paired closed-loop runs must restart from the same initial state.
Temporary unselected request files are removed only from the new run directory;
episode outcomes and partial indexes remain if collection aborts.

## 3. T0 and E1

```bash
python -m benchmarks.libero.research t0 \
  --ckpt-dir CHECKPOINT_DIRECTORY --ckpt-name CHECKPOINT.safetensors \
  --split experiment/manifests/split.json --bank experiment/state_bank/discovery \
  --seeds 42 43 44 --output experiment/diagnostics/t0_discovery
python -m benchmarks.libero.research plan --experiment e1 --layers ACTUAL_LAYER_COUNT \
  --split experiment/manifests/split.json --output experiment/manifests/e1.json
python -m benchmarks.libero.research smoke \
  --ckpt-dir CHECKPOINT_DIRECTORY --ckpt-name CHECKPOINT.safetensors \
  --split experiment/manifests/split.json --bank experiment/state_bank/discovery \
  --plan experiment/manifests/e1.json --method-id E1_L0_S0_action \
  --seeds 42 --output experiment/diagnostics/e1_smoke
python -m benchmarks.libero.research replay \
  --ckpt-dir CHECKPOINT_DIRECTORY --ckpt-name CHECKPOINT.safetensors \
  --split experiment/manifests/split.json --bank experiment/state_bank/discovery \
  --plan experiment/manifests/e1.json \
  --t0 experiment/diagnostics/t0_discovery/t0_certificate.json \
  --seeds 42 --output experiment/diagnostics/e1_screen
```

T0 uses 20 observations × exactly three distinct seeds. It compares raw repeated
baseline and gate-one output using elementwise repeated error plus one
dtype-scaled epsilon; it also requires identical converted gripper commands.
One of the seeds must match collection; that branch must reproduce the saved
baseline chunk. The bank's recorded model/source/configuration must match.
The report includes absolute/relative errors. A failure is a stop, not a license
to increase a blanket tolerance. CPU tests separately verify the explicit
attention formula, both directions, prefix protection, value-only perturbations,
unselected rows and request isolation. Batch screening requires a passing T0
certificate bound to the **same model/source/bank/protocol**. Run T0 separately
for the Calibration bank. `smoke` runs one chosen method/state/seed without a
certificate; its output is diagnostic only and is not a T0 pass.

E1 emits 24 gate-zero configurations (four continuous layer groups × three
continuous step groups × two directions), B0, official default B1, and B2 at
4/6/8 steps. Discovery defaults to episodes 0–4, at most 120 states. Each method
includes a fresh paired reference generation, so budget for these extra calls.
Start with the action direction; to narrow the configuration list, author a
new plan JSON with the desired methods, `selected_split`, `experiment`,
`fitted_split=discovery`, and freeze it as described below. Calibration plans
use `plan --selected-split calibration` and at most ten episodes/task.

Individual layer IDs, arbitrary step IDs, gate values 0/0.5/0.75/1, and token
lists are supported. For fine selection, record at most 20 first-round configs
in a custom plan. `plan --experiment retention` consumes a sealed Discovery
ranking with `positions` in most-important-first order and emits matched
25%/50%/75% ranked/uniform/five-random retention sets. `--experiment cumulative`
emits cumulative low-ranking deletion sets. Combinations execute the exact
position set, not its layer/step Cartesian product. B4 is available via custom
methods with `cache: {cosine_threshold: 0.98, max_consecutive_skips: 1}`; choose
thresholds only on Calibration. Official cache code is not modified.

## 4. E2 selection, features and paired perturbations

Choose representative positions **from Discovery E1**, not test results. Author
an unsealed selection JSON with the following fields:

```json
{
  "fitted_split": "discovery",
  "layer_ids": [0, 1],
  "step_ids": [1, 5, 9],
  "layouts": {"0": {}, "1": {}},
  "state_ids_by_split": {"discovery": [], "calibration": []}
}
```

Replace layer/step IDs with the chosen actual IDs, copy the two actual layout
objects from E1 result `layouts`, and freeze 24 state IDs per intended split:
one from each progress stage for each of eight tasks. Empty objects/lists above
are **schema illustrations, not runnable defaults**. Record the source E1 run
and rationale as additional fields. A ranking artifact instead has `positions`
covering every layer/step pair exactly once.

```bash
python -m benchmarks.libero.research freeze --input selection.json \
  --split experiment/manifests/split.json --output experiment/manifests/selection.json
python -m benchmarks.libero.research features \
  --ckpt-dir CHECKPOINT_DIRECTORY --ckpt-name CHECKPOINT.safetensors \
  --split experiment/manifests/split.json --bank experiment/state_bank/calibration \
  --selected-split calibration --selection experiment/manifests/selection.json \
  --t0 experiment/diagnostics/t0_calibration/t0_certificate.json \
  --seeds 42 --output experiment/diagnostics/calibration_features
python -m benchmarks.libero.research plan --experiment e2 --layers ACTUAL_LAYER_COUNT \
  --split experiment/manifests/split.json --selection experiment/manifests/selection.json \
  --calibration experiment/diagnostics/calibration_features/calibration.json \
  --output experiment/manifests/e2.json
```

Feature collection runs only unperturbed gate-one generations and stores group
norm/change statistics; it does not fit an action rule. The frozen floor is
`max(median(Calibration group norms) * 1e-6, float32.tiny)`. Generate Discovery
feature statistics with the analogous `features --selected-split discovery`
command and its own bank/T0 certificate.

E2 defaults to two layers × three steps × up to eight equal-size future groups
× 24 states × positive/negative perturbations at epsilon 0.03. Run it with the
same `smoke`/`replay` commands and the E2 plan. Groups may be fewer than eight;
inspect the plan rather than assuming 2,304 calls. Baseline generations are
additional. Only expand amplitudes or change perturbation types in an explicitly
new frozen plan. Stale replacement at step zero is invalid. No decoded-video,
black-camera, attention-mass or PCA proxy is treated as an oracle for task quality.

## 5. Results, validity and tests

```bash
python -m benchmarks.libero.research summarize \
  --runs experiment/diagnostics/e1_screen --output experiment/tables/e1
python -m benchmarks.libero.research summarize \
  --runs experiment/diagnostics/e2_screen --output experiment/tables/e2 \
  --features experiment/diagnostics/discovery_features/features.json \
  --calibration experiment/diagnostics/calibration_features/calibration.json
python -m pytest -q tests/test_research_interventions.py tests/test_research_protocol.py \
  tests/test_research_integration.py tests/test_research_analysis.py
```

Metrics exclude inactive unified dimensions and distinguish normalized active
error, native EEF10 error, converted translation/rotation **command** differences
and gripper disagreement, for both the execution prefix and full chunk. Rotation
command difference is not a physical orientation angle. Local action-velocity
responses and both sigmas are saved. Summary tables first average within episode;
multiple states/seeds do not become independent episode replicates. E2 ranks
groups within state/seed/layer/step, then writes episode-level association means.
Constant rankings produce null Spearman, not invented zero correlations.

Plots are emitted when matplotlib and appropriate data are available; raw tables
are always retained. No SR confidence interval or success–latency Pareto curve
is produced without actual closed-loop/benchmark data. `diagnostic_server_ms`
includes extra instrumentation and host synchronization and **must never be
used as a deployment speedup**.

Only `OSError`/connection/timeout failures during fixed-state replay get at most
two retries; each rejected attempt is logged. Configuration errors stop the run.
Numerical policy failures are retained as policy failures and stop screening;
they are not automatically retried. Other unexpected runtime errors stop with
their original error recorded, rather than guessing their cause. Collection
does not retry from an unverified mid-episode snapshot. Partial runs are not
silently turned into complete runs, and output directories are never resumed
or overwritten implicitly.

On a machine without functional CUDA, preflight/CPU tests remain usable; model,
simulator, T0, H1/H2 and speed conclusions remain `not_run` until genuinely run.
# E2 short-trajectory exploratory protocol (v1)

`bash scripts/research_alpha_libero_e2_exploratory.sh` reuses the seed-43
development banks without modifying their manifests. `GPU=4` selects physical
GPU 4 by UUID. `PREPARE_ONLY=1` freezes and checks inputs without loading CUDA.

The explicit `e2_progress_fallback_v1` protocol keeps all eight tasks and selects
three distinct states per task. It first chooses the closest state to each
available 20/50/80% progress target, breaking ties by episode/request/state ID.
Missing bins are filled from another episode when available, then closest to
50%, with the same deterministic ties. Coverage records retain the actual bins
and missing bins. These are progress proxies, not contact labels. This is an
exploratory protocol amendment, not complete early/middle/late coverage.
The original strict E2 validator remains the default.

Layers 0/29 and steps 1/5/9 are frozen from Discovery E1; saved Discovery layouts
and evidence hashes are recorded. Calibration feature norms determine the floor;
both splits then receive paired positive/negative epsilon=0.03 interventions.
Only the selected 24 states per split are used for feature collection.

Source-only transition reviews permit only the enumerated research entrypoint,
feature collector, exploratory helper and its tests to change. All other model
identity fields must match, including assets, configuration, backend and runtime.
Original bank/model hashes remain unchanged; each run records the review and
requires fresh T0 certificates before replay. No old T0 certificate is reused.
The script stops on any error and writes progress to `RUN/launcher.log`.
No held-out tasks, closed-loop success or actual speedup are established here.
