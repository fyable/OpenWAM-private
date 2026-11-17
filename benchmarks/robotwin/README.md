# RoboTwin 2.0 Evaluation

## One-command Native / Joint comparison

On the current server, an offline local installation can be restored with:

```bash
bash scripts/setup_alpha_robotwin.sh
```

This copies the readable simulator packages and precompiled cuRobo extensions
from the existing `/data2/zhibo/LHN/wam` installation into `.venv-robotwin`, and
copies the verified RoboTwin code from `/data1/cuihanyun/RoboTwin` into
`third_party/RoboTwin`. It uses about 10 GB of local disk. It does not modify the
source environments. Large objects/background textures remain read-only links
to `/data2/vla-data/robotwin/assets`; the Aloha robot files are copied locally and
their stale absolute cuRobo paths are rewritten. Keep the shared asset directory
available. This is an offline restoration, not a portable clean-install script;
source overrides are listed in `scripts/setup_alpha_robotwin.sh`.

The installed stack uses PyTorch 2.7.1+cu128 and warp-lang 1.9.0, rather than the
released checkpoint's verified stack listed below. CPU imports, all five cuRobo
compiled modules, and `pip check` have been checked, but these checks do not
establish GPU rendering or episode correctness. Always run diagnostic on the
host before full evaluation.

The setup writes `outputs/robotwin/environment.sh` and
`outputs/robotwin/environment_check.json`. The comparison shell launcher loads
the environment defaults automatically without overriding explicitly supplied
variables, so after setup the current server command is simply:

```bash
GPU=0 bash scripts/compare_alpha_robotwin.sh diagnostic
# After diagnostic succeeds:
GPU=0 bash scripts/compare_alpha_robotwin.sh full
```

The local comparison launcher runs **Native N10/E8 → Native N4/E8 → Joint
stage-parallel N4/E8** sequentially, starting and stopping its own policy servers.
It does not require a Labtasker installation or service. Install RoboTwin and its
task assets first using the environment setup below. This launcher does not
install simulator dependencies or download assets automatically.

```bash
cd /home/zhijian/OpenWAM-main

ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/robotwin/bin/python \
GPU=0 \
COMPARE_ROOT="$PWD/outputs/robotwin/compare_joint_clean" \
bash scripts/compare_alpha_robotwin.sh full
```

Replace both RoboTwin paths with your installed environment. The default model
Python is `.venv-openwam/bin/python`; the default checkpoint directory is
`/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full`, using its highest numbered
`checkpoint_step_*.safetensors`. Override them with `SERVER_PYTHON` and `CKPT_DIR`.
Only use GPUs you are authorized to use. `SIM_GPU` defaults to `GPU`; set it to a
different available GPU if the model and simulator cannot fit on the same card.
`PORT` defaults to 8940 and `SERVER_START_TIMEOUT` to 900 seconds.

**Run `diagnostic` first**, with the same environment variables and a different
`COMPARE_ROOT`, to check your actual simulator/assets and the experimental Joint
adapter. It evaluates `adjust_bottle`, `open_laptop`, and `stack_blocks_two`, five
episodes each, across all three methods (45 episodes total). RoboTwin Joint GPU
compatibility is not established by the launcher's CPU tests.

Modes:
- `plan`: print the protocol without creating output directories or launching jobs.
- `diagnostic`: the 45-episode check above; default output
  `outputs/robotwin/compare_joint_diagnostic`.
- `full`: clean only, all 50 tasks, 100 episodes per task per method (15,000 total);
  default output `outputs/robotwin/compare_joint_clean`.
- `summary`: read saved results and regenerate `comparison.json` / `comparison.md`,
  without launching jobs; exits nonzero while any task/method is incomplete.

### Four-arm Joint / isolated comparison

```bash
bash scripts/compare_alpha_robotwin_joint_isolated.sh plan
bash scripts/compare_alpha_robotwin_joint_isolated.sh preflight
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh diagnostic
```

The default runs `adjust_bottle`, five shared manifest episodes per method (20 total):
Joint Native N10, Joint parallel N10, isolated Native N10, and isolated original
ParaFlow N10. Joint uses `/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full`;
isolated uses step 9890 under
`/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port/assets/openwam`.
Override with `JOINT_CKPT_DIR` and `ISOLATED_CKPT_DIR`. Preflight checks the original
checkpoint mask; it does not silently treat mutual weights as isolated training.
Both checkpoint identities and original ParaFlow source hashes are saved for resume.

Set `TASKS=adjust_bottle,open_laptop EPISODES=5` for a larger diagnostic.
`full` runs 50 tasks, 100 episodes, four methods (20,000 episodes).
Outputs default to `outputs/robotwin/joint_isolated_diagnostic` or
`outputs/robotwin/joint_isolated_full`; use `COMPARE_ROOT` to select a fresh directory,
or reuse the unchanged directory to resume. `summary` regenerates `comparison.md`
and `comparison.json`. All four arms reuse the exact same eligible scene manifests.
Compare optimization effects **within each checkpoint**: mutual generates future-video
latents, isolated uses only current-frame conditioning, so cross-family timing is not
an equivalent-workload algorithm speedup. Logged experimental request timings are
diagnostic (including retries), not paired replay speed measurements; native full-joint
request timings are not currently captured in that summary. Numerical gates stay strict.
GPU compatibility and closed-loop success remain to be validated on the host.

### Mutual Joint execution optimization

Three opt-in execution backends keep the same N10/H32/E8 sampler, mutual mask,
noise/history alignment, and FP32 scan: `legacy` (default), `prepared` (request-local
text/proprio conditioning projections), and `prefix` (prepared conditioning plus
clean-first-frame per-layer K/V prefill). Future video and action remain mutually
coupled. Prefix dynamic queries keep original absolute RoPE positions and attend
the original ordered full K/V; the complete video head still produces all frames.
Cached visual/context values never survive an observation. Shared checkpoint and
ParaFlow source directories are not modified.

```bash
# Five arms, three tasks, five matched episodes each = 75 episodes.
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh execution-ablation

# Once the short quality check passes: legacy versus prefix, 60 paired scenes.
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh execution-regression

# One command: ablation, separate real RoboTwin request capture, validated replay/profile.
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh execution-all
```

The five-arm ablation includes Joint Native, Joint legacy/prepared/prefix parallel,
and original isolated ParaFlow as a reference. Defaults use `adjust_bottle`,
`open_laptop`, `stack_blocks_two`; `TASKS` and `EPISODES` override diagnostic sizes.
Default ablation output: `outputs/robotwin/joint_execution_ablation/`.
Default regression output: `outputs/robotwin/joint_execution_regression/`.
`execution-all` chooses a timestamped root; set `COMPARE_ROOT` explicitly to resume.
It runs 75 quality episodes plus 15 capture episodes by default; the larger
120-episode regression remains a separate explicit command.
Do not reuse directories created by older code/protocol versions.

Separate capture/replay is available:

```bash
GPU=6 bash scripts/compare_alpha_robotwin_joint_isolated.sh capture
GPU=6 TRACE="$PWD/outputs/robotwin/joint_execution_capture/observations.npz" \
  bash scripts/compare_alpha_robotwin_joint_isolated.sh replay
```

Capture records real E8 generation inputs and saves a provenance sidecar; replay
rejects LIBERO/unverified traces and requires an exclusive GPU. With defaults it
validates 48 warmup + 120 eligible warm requests, compares independent multi-request
trajectories/executed actions, then separately times identical legacy-owned candidate
histories with cloned RNG in seeded shuffled order. Validation can be expensive:
frozen-candidate video/action velocities and batch-versus-independent partitions are
checked separately at rtol/atol 0.05. Any nonfinite value is fatal. Replay saves
`validation.json[l]`, `requests.jsonl`, `timing.md/json`, `profile.md/json`, and failure artifacts.
`ALLOW_PARITY_FAILURE=1` applies only to replay diagnostics, never establishes equivalence.
If capture has too few requests, increase task/episode coverage in a new root.

`WARMUP`, `MEASURE`, `PROFILE_SAMPLES` (default 48/120/8) control replay. Profiling is
a separate pass, with CUDA events for conditioning, prefix prefill, state preparation,
joint blocks, heads, trajectory update/constraints and VAE encoding where available;
Chrome traces additionally expose CPU/kernel behavior. `PROFILE_SAMPLES=0` disables
profiling. Peak allocated/reserved memory is recorded; sampler histories coexist,
and allocator-reserved memory depends on execution order. Do not interpret it as
an isolated single-process model footprint.

Acceptance: at least 10% lower paired prefix P50 and no P95 regression, strict numeric
checks passed, and no success-count regression against same-run legacy in both short
and 60-scene quality checks. Reports list per-episode quality changes. Passing replay
alone never approves closed-loop quality. Native full-request timings are now opt-in
for these ablations, but closed-loop timing remains diagnostic, not paired speedup.
The default stays `legacy` regardless of results; no GPU speedup is claimed from CPU tests.
Direct deployment uses `inference.joint_execution_backend=prepared|prefix` with
`--sampler joint-paraflow`, compile/cache/decode disabled and CFG=1. Unsupported
conditions fail explicitly rather than silently dropping video or changing the mask.

All methods use synchronous E8 execution/denoising, CFG=1, and disabled compile,
DiT cache and video decoding. The checkpoint's training observation/action
configuration is retained, with the bundled end-effector RoboTwin policy adapter.
The client uses unseen instructions and seed-0 candidate scanning. For each task,
the launcher first builds an expert-eligible scene manifest and then reuses its
exact accepted seeds and instructions across all three methods. This preliminary
expert scan also consumes simulator time; the full campaign takes many GPU-hours.
The `labtasker` word in evaluator subprocess commands selects an existing wrapper
operation, not a Labtasker server dependency.

Results and simulator runtime files stay under `COMPARE_ROOT`, not in the
checkpoint or upstream RoboTwin directory. Each method has `servers/`,
`attempts/<task>/`, and validated `results/<task>.json`; shared manifests live in
`manifests/`. Task logs and `progress.json` are under each attempt. A failure stops
the campaign, cleans up only owned processes, and writes an incomplete summary.
Re-run the **same command with the same `COMPARE_ROOT`** to skip completed tasks;
an interrupted/incomplete task is repeated from its first episode. Protocol,
checkpoint metadata/configuration, simulator sources and local code hashes must
match; use a fresh root after changing them. Weight contents are not rehashed:
keep weights and assets immutable. Simulator package versions and assets are not
fully fingerprinted, so do not change the environment between resumed attempts.
Concurrent campaigns in one root and occupied policy ports are rejected.

```bash
COMPARE_ROOT="$PWD/outputs/robotwin/compare_joint_clean" \
bash scripts/compare_alpha_robotwin.sh summary
```

Reports show overall and per-task success counts and missing tasks. Partial rates
cover only complete tasks, not the full benchmark. Joint cold/warm request timing
is extracted from all server attempts (including retries); it is **not** a paired
speedup measurement against Native. No full GPU campaign is launched by setup or
unit tests.

## Other launchers

For more efficient parallel evaluation with finer-grained control, use the
[Labtasker-based evaluation](LABTASKER.md).

**If you prefer the traditional launchers, follow the instructions below.**

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **RoboTwin client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere.

Commands below assume conda at `/path/to/miniconda3` and the RoboTwin checkout at `/path/to/RoboTwin` — substitute your actual paths.

## 1. Environment Setup

Build the RoboTwin env separately from OpenWAM's (do not mix them):

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
# follow RoboTwin's official installation guide → conda env "robotwin" + task assets

/path/to/miniconda3/envs/robotwin/bin/pip install websockets pyyaml   # client deps OpenWAM needs
```

- `ROBOTWIN_PATH` → the checkout; `ROBOTWIN_PYTHON` → `/path/to/miniconda3/envs/robotwin/bin/python`. Every eval command needs both (`single_eval.sh` refuses to start without them).
- Headless nodes normally work as-is: SAPIEN renders offscreen through its bundled Vulkan loader, no X server needed. Only if you hit a display/EGL error at SAPIEN import, prefix the command with `xvfb-run -a`.

**Verified versions.** The adapter (`eval_policy_wrapper.py`) loads RoboTwin's `script/eval_policy.py` by path and patches it, so it is tied to the upstream layout. The combination below is what the released checkpoints were evaluated with; other versions may work but are unverified (the wrapper prints a warning when the RoboTwin commit differs, `ROBOTWIN_SKIP_VERSION_CHECK=1` silences it).

| Component | Version |
|---|---|
| RoboTwin | commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` (2026-04-19) |
| Python (RoboTwin env) | 3.10 |
| SAPIEN | 3.0.0b1 |
| cuRobo | v0.7.8, built into `<RoboTwin>/envs/curobo` by RoboTwin's `script/_install.sh` |
| warp-lang | 1.13.0 |
| mplib | 0.2.1 |
| torch (RoboTwin env) | 2.4.1 (the policy server uses OpenWAM's own env) |

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-RoboTwin-Full
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full`. (`...-Clean2Random` is the OOD variant trained on clean data only; any checkpoint you trained yourself works the same way.) Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full
```

WebSocket port 8848 by default (`--port` to change). Keep it running.

## 3. Run the Evaluation

Single task — args are `task_name task_config ckpt_setting gpu_id [port] [host]`:

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/miniconda3/envs/robotwin/bin/python \
bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean openwam 0
```

All 50 tasks (`-m` mode, `-n` run name, `-d` checkpoint dir; tasks = names, comma lists, `all`, or a file):

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/miniconda3/envs/robotwin/bin/python \
bash benchmarks/robotwin/multi_eval.sh -m demo_clean -n run1 \
  -d assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full all
```

`benchmarks/robotwin/policy_config.yml` must match the checkpoint: `action_type: ee` + `state_dim: 20` (end-effector, the released checkpoints) or `action_type: qpos` + `state_dim: 14` (joint-space).

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Start the server first; the client health-checks it for up to 300 s, then aborts. Remote server: pass `[host]`/`[port]` (single) or `--host`/`--port` (multi); defaults `127.0.0.1:8848`.
- `task_config` ∈ {`demo_clean`, `demo_randomized`}; `ckpt_setting` is only a label in RoboTwin's result filenames; `-d` is used for log placement — weights never load client-side.
- `ROBOTWIN_TEST_NUM=5` caps episodes per task for a quick smoke run (default 100/task; the full `all` run takes many GPU-hours per mode).
- Don't call RoboTwin's `eval_policy.py` directly — go through `eval_policy_wrapper.py`. Its startup messages are expected; each one works around a known crash in the verified stack:
  - `prewarmed CUDA/Curobo before SAPIEN import` — cuRobo must initialise CUDA before SAPIEN's Vulkan context exists, otherwise the first plan segfaults.
  - `patched warp.torch compatibility namespace` — cuRobo v0.7.8 still calls `warp.torch.*`, which warp ≥ 1.x no longer ships; the wrapper aliases the new top-level functions.
  - `SAPIEN EGL ICD: …` (from `single_eval.sh`) — SAPIEN's EGL probe crashes on images without `/usr/share/glvnd/egl_vendor.d`; the script points it at SAPIEN's bundled ICD instead.
  - `RoboTwin commit=… eval_policy.py sha256=…` — provenance of the RoboTwin code actually loaded; a `WARNING` follows if the commit differs from the verified one or files under `script/ envs/ task_config/ policy/` are locally modified. Check out the verified commit before debugging anything else.
- If cuRobo planning itself fails, set `ROBOTWIN_ENABLE_PLANNER_FALLBACK=1` to plan with `mplib_RRT` instead (slower, results not comparable to the tables below).
- Read-only checkout? Set `ROBOTWIN_RUNTIME_ROOT` to a writable dir. Per-task step limits: `benchmarks/robotwin/step_limits.yml` (all commented out by default; read once at startup).
- Results: RoboTwin's native `eval_result/` inside the checkout (or runtime root); `multi_eval.sh` also tees per-task logs under `<ckpt_dir>/robotwin_eval_logs/…`, records the run there (`run.env` — shell-quoted, `source`-able — plus verbatim copies of the `policy_config.yml` and `step_limits.yml` used), and prints each task's `Success rate`.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

Evaluation protocol for the OpenWAM-α rows (reproduce with the commands in §3): released checkpoint served with `scripts/deploy.sh` defaults from `configs/deploy.yaml` — `denoise_steps: 10`, `denoise_mode: sync`, DiT velocity cache **on** (`cosine_threshold: 0.99`, `max_skips: 3`), `torch.compile` on — 100 episodes per task at seed 0, RoboTwin's upstream per-task step limits (`step_limits.yml` empty), and the versions in §1. Changing any of these (in particular disabling the DiT cache) changes the numbers.

**RoboTwin2.0-Clean2Random** (fine-tune on clean only; OOD probe):

| Method | Type | Clean | Randomized | Avg |
|---|---|---:|---:|---:|
| StarVLA | VLA | 46.5 | 3.2 | 24.9 |
| GR00T-N1.7 | VLA | 43.6 | 20.7 | 32.2 |
| X-VLA | VLA | 68.0 | 20.9 | 44.5 |
| Spatial Forcing | VLA | 77.2 | 26.7 | 52.0 |
| ABot-M0 | VLA | 70.7 | 36.0 | 53.4 |
| π₀.₅ | VLA | 70.7 | 46.0 | 58.4 |
| GigaBrain-0.7 | VLA | 66.8 | <u>67.9</u> | 67.4 |
| Qwen-RobotManip | VLA | <u>84.7</u> | **69.4** | **77.1** |
| AHA-WAM | WAM | 64.3 | 3.2 | 33.8 |
| Fast-WAM | WAM | 77.8 | 1.9 | 39.9 |
| X-WAM | WAM | 70.0 | 25.8 | 47.9 |
| 4D-WAM | WAM | 81.5 | 41.8 | 61.7 |
| **OpenWAM-α** | WAM | **89.4** | 48.7 | <u>69.0</u> |

**RoboTwin2.0-Full** (fine-tune on clean + randomized; ID probe):

| Method | Type | Clean | Randomized | Avg |
|---|---|---:|---:|---:|
| X-VLA | VLA | 72.80 | 72.84 | 72.82 |
| π₀.₅ | VLA | 82.70 | 76.80 | 79.75 |
| ABot-M0 | VLA | 86.06 | 85.08 | 85.57 |
| Qwen-VLA | VLA | 86.10 | 87.20 | 86.65 |
| StarVLA | VLA | 88.18 | 88.32 | 88.25 |
| Galaxea G0.5 | VLA | 93.70 | 92.80 | 93.25 |
| Qwen-RobotManip | VLA | 93.70 | <u>94.00</u> | <u>93.85</u> |
| Motus | WAM | 88.66 | 87.02 | 87.84 |
| Fast-WAM | WAM | 91.90 | 91.80 | 91.85 |
| LingBot-VA | WAM | 92.93 | 91.55 | 92.24 |
| ImageWAM | WAM | 93.20 | 93.56 | 93.38 |
| LingBot-VA 2.0 | WAM | <u>93.80</u> | 93.40 | 93.60 |
| ABot-M0.5 | WAM | **94.00** | **94.20** | **94.10** |
| **OpenWAM-α** | WAM | 93.74 | 93.46 | 93.60 |
