# Chunk pipeline latency experiment

Run `GPU_LIST=0,2,4,6 bash scripts/profile_chunk_pipeline.sh` from the repository root.
The experiment loads four GPU-resident replicas for each phase, with no CPU offload,
compile, DiT cache, pacing sleep, or changes to the N10/H32/E8 round-barrier algorithm.
It runs control20, instrumented20 (four suites, task0, five initial states), then
one spatial/task0 episode for detailed tracing if the profiler capability probe passes.
The one-episode trace uses the scheduler's `--smoke` protocol. To recover an interrupted
trace phase without rerunning the completed pair, set `PHASES=trace RUN=<existing directory>`.
Prior launcher, source fingerprints and initial GPU snapshot are archived automatically.
Phase manifests still validate any resumed evaluation.
Existing reproduction results are not overwritten. Do not edit sources during a run.

## Instrumentation

- `OPENWAM_CHUNK_DIAGNOSTICS=1` enables JSONL wall-clock and CUDA-event spans.
  `OPENWAM_CHUNK_DIAGNOSTICS_DIR` is the output directory. Both must be set to enable collection.
- `OPENWAM_CHUNK_TRACE=1` additionally enables all-thread PyTorch CPU/CUDA profiling.
  It requires diagnostics. Sampling starts at chunk16 and stops after the chunk56 barrier,
  covering four steady head-worker rotations and background dependencies.
- Default is off. No new per-region device synchronization is added. Event results
  are queried after existing completion points; episode cleanup drains remaining events.
- Worker records carry pipeline episode, server episode/action step, target round,
  worker, logical device, physical GPU, condition source, chunk and stage where applicable.
  Background target round is the next generation request; online round is the current request.
- CPU monotonic timestamps permit alignment across the local processes. CUDA events
  are only compared within a device/stream. Their spans include host submission gaps;
  they are not sums of GPU kernel execution time.
- `_call` transfer regions include casting and possible waits for earlier GPU work.
  Pure memcpy duration is available only from a complete detailed CUDA activity trace.
  PIL resize outside the tensor helper is part of the enclosing preparation span.

## Outputs and interpretation

`diagnostics.json` contains paired results, trial differences, instrumentation overhead,
per-worker segment distributions, barrier-window partitions and trace coverage.
`REPORT.md` provides headline statistics; `timeline.html` shows inclusive host intervals.
Each phase keeps manifests, original client/server logs and diagnostic JSONL.
The root directory contains source fingerprints, profiler capability results, and
one-second GPU utilization/process monitoring. Inspect external processes on the selected
cards before treating timing differences as instrumentation overhead.

The baseline is the control run, not the profiler run. Differences above 5% in either
steady client P50 or P95 are marked as perturbation. Inspect paired success, policy steps
and initial states before attributing any behavioral changes to normal variation.
Warm-up excludes the first three chunks per episode. Environment-step sums exclude
policy calls; the eight-action execution window includes buffered policy calls and transport.
Terminal partial chunks do not produce a complete eight-action window.

Never add inclusive regions or parallel worker times, subtract independent medians,
or classify a synchronous `.cpu()` call as pure PCIe copy. Critical-path categories
partition each observed barrier window first, then calculate distributions.
Inapplicable I2V/VACE dispatch spans are not evidence of actual encoder work;
`vae_encode`, `vae_encode_vace`, and applicability records identify the executed paths.
A profiler that misses threads/devices/memcpy is explicitly incomplete; no pure-transfer
bottleneck conclusion may be drawn from it. No profiler dependency is installed automatically.

Rank follow-up optimizations by exposed critical-path time and supporting GPU activity:
tensor transfers/PIL elimination, decode/encode redundancy, noise initialization,
or denoise/synchronization execution. These experiments diagnose only; all optimizations
need a separate controlled A/B and action/success validation.
