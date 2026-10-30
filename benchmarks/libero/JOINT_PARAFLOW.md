# Joint ParaFlow: experimental Alpha LIBERO adapter

This opt-in sampler uses the joint self-attention Alpha checkpoint without
training or modifying weights. Native inference remains the default.

## Mechanism and limits

- H32/E8, synchronous joint denoising, N4 by default, single-GPU stage batching.
- The nine-frame video window has three VAE latent cells: one observed clean
  prefix and two predicted cells. Each future cell corresponds to 16 actions.
- Two video histories alternate by request. The first two requests bootstrap
  their phases with N sequential joint forwards each. Starting at request three,
  one forward batches N stage candidates as independent samples. This is **not**
  an exact one-step diffusion model or a claim of N-fold end-to-end acceleration.
- Actions align from the previous request by eight positions. Video aligns
  from the same phase two requests earlier by one whole latent cell. The current
  observed prefix replaces every stage prefix. E10 is deliberately rejected.
- Action updates use a causal affine recurrence; video uses an Euler prefix.
  The method references ParaFlow, but all alignment, noise coupling and updates
  execute in `openwam/deploy/joint_solver.py` within this repository. No external
  ParaFlow installation, source checkout or `PYTHONPATH` setting is required.
- History and scans are FP32; model inputs use the architecture dtype. Inactive
  action coordinates follow their analytic noise trajectory and finish at zero.
- Per-episode reset clears histories, phase state and RNGs. Prompt, seed, grid,
  shape, dtype, device or active-mask changes also invalidate history.
- Requires Wan2.2 TI2V, mutual attention, CFG=1, video decoding disabled,
  compilation disabled and DiT velocity cache disabled. The adapter passes
  scheduler training-time units to both joint branches, not normalized sigmas.

Batch=4 can substantially increase activation memory. Run the replay and smoke
checks before any larger evaluation. The checkpoint is read-only. The reference
repository is not a runtime dependency. Server logs and evaluation manifests record local core
source paths and SHA256 hashes; resume refuses changed core identities.

## Run

From the OpenWAM repository root, with working NVIDIA driver access:

```bash
export CKPT_DIR=/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO
GPUS=0 bash scripts/eval_alpha_libero.sh joint-smoke
GPUS=0 bash scripts/eval_alpha_libero.sh joint-diagnostic
```

Smoke is one task/episode. Diagnostic is task 0 in each of spatial, goal,
object and long, with five trials each: **20 episodes**, not the official
full evaluation. Existing settling, seed and step-limit checks remain enforced.
Each launcher invocation gets a separate timestamped output directory; use
`OUTPUT_DIR` to select one explicitly. A larger `joint-eval` mode exists but is
not launched automatically. Results go into that directory's `summary.json`,
`manifest.json` and server/client logs.

Do not overwrite `outputs/libero/alpha_full`. Its historical native N10/E10
result is 1984/2000 (99.2%); E8 diagnostics are not directly protocol-comparable
and must not be presented as a reproduction of that baseline.

## One-command comparison

```bash
# Print the protocol without starting any GPU jobs.
bash scripts/compare_alpha_libero.sh plan

# All three methods, 20 episodes each, for checking the complete workflow.
COMPARE_ROOT="$PWD/outputs/libero/compare_joint_diagnostic" \
  bash scripts/compare_alpha_libero.sh diagnostic

# Native N10/E8, Native N4/E8, Joint N4/E8; 2000 episodes per method.
COMPARE_ROOT="$PWD/outputs/libero/compare_joint_full" \
  bash scripts/compare_alpha_libero.sh full

# Read existing outputs only; never starts model inference.
COMPARE_ROOT="$PWD/outputs/libero/compare_joint_full" \
  bash scripts/compare_alpha_libero.sh summary
```

`GPUS`, `RENDER_GPUS`, `SERVER_PYTHON`, `LIBERO_PYTHON`, `CKPT_DIR` and
`REPLICAS_PER_GPU` are configurable environment variables. Defaults use GPU 0,
one replica, the existing local environments and the checkpoint under
`/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO`. All methods run sequentially, with
compilation, decoding and DiT cache disabled. A failure stops subsequent methods.
Rerun the same mode with the same `COMPARE_ROOT` to resume; the scheduler keeps
completed tasks and checks its protocol/source signature. Diagnostic and full
runs must use different roots. The final `comparison.json` retains per-suite
and overall success rates, completion state and joint cold/warm timing summaries.
Missing or incomplete results cause a nonzero summary exit, not a success claim.
Never reuse the earlier external-core experiment directories: source identity
has intentionally changed. Old external-core results remain historical evidence,
not verification of the local implementation.

For the paired replay, set `TRACE` to a real E8 observation NPZ and run:

```bash
TRACE=/path/to/fixed_e8_observations.npz \
COMPARE_ROOT="$PWD/outputs/libero/compare_joint_full" \
  bash scripts/compare_alpha_libero.sh replay
```

This verifies a real-model warm batch first and refuses to continue unless parity
passes. It then times all three methods separately. Replay outputs refuse
overwrite; choose a fresh root for a new replay attempt. It never synthesizes
observations or runs native closed-loop episodes to capture them.

## Fixed-observation replay

Use an existing observation trace, without running native closed-loop episodes.
Create an NPZ with `allow_pickle=False`-compatible arrays:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `images` | uint8 `[N,H,W,3]` | Engine-ready RGB images after the normal observation preprocessing/view composition |
| `prompts` | Unicode `[N]` | Task text, constant within each episode |
| `states` | finite numeric `[N,D]` | Raw deploy proprio, before engine normalization |
| `episode_starts` | bool `[N]` | First entry true; reset boundaries |

Observations must be sampled at **eight-action intervals**, with at least three
requests per episode to measure warm behavior. Do not substitute arbitrary
consecutive camera frames. No trace is synthesized or claimed to be a simulator
capture by this implementation.

```bash
for spec in native:4 native:10 joint-paraflow:4; do
  method=${spec%:*}; steps=${spec#*:}
  .venv-openwam/bin/python -m benchmarks.libero.replay_joint \
    --observations /path/to/fixed_e8_observations.npz \
    --ckpt-dir "$CKPT_DIR" --sampler "$method" --steps "$steps" \
    --output "outputs/joint_replay/${method}_n${steps}.json"
done
```

Replay disables compilation, DiT cache and decoding for all methods. It records
full engine-request P50/P95, actual forward batches, per-request peak allocated
CUDA memory, cold/warm classification, joint denoising time and source identity.
Full request time includes prompt/image preprocessing, VAE encoding and output
normalization but excludes transport and simulator time. Native reports do not
claim a separately instrumented denoising time. Compare identical NPZ hashes;
keep cold startup and warm measurements separate. GPU/kernel startup is included
in the first recorded request. Output overwrite is refused.

Before timing runs, repeat the joint replay with `--verify-batch` and a separate
output file. This compares the first warm batch against independent real-model
forwards at each stage's own video/action times. Both modalities must pass
`torch.testing.assert_close` (default BF16-oriented `--rtol 0.05 --atol 0.05`);
the report retains maximum absolute errors and tolerances. Inspect these errors,
not just the pass label. If no warm batch occurs, parity remains `not_run`.
Verification adds sequential forwards, so **do not use its timings** for speed
comparisons. Run without this flag for latency measurements.

## Validation status (2026-10-27)

- CPU adapter tests pass: stage batching vs independent fake-model calls,
  local scan/alignment core, two-phase bootstrap, prefix replacement,
  inactive coordinates, deterministic reset, unbatched proprio and fail-fast.
- Deployment, scheduling, execution, representation and manifest regressions
  pass; the optional labtasker test is skipped because labtasker is absent.
- Real-checkpoint batch-vs-sequential GPU numerical parity: **not_run**.
- Fixed-observation GPU latency and memory comparison: **not_run**.
- LIBERO smoke and 20 diagnostic episodes: **not_run**.
- No native closed-loop rerun or full GPU campaign is launched. CUDA initialization
  in the current sandbox fails with error 304, so no speedup or success rate is
  asserted for this sampler. Passing CPU contracts is not a performance gate.
