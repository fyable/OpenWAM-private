# Original ParaFlow / Joint optimization benchmark

## Run

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
- Sweeps official modes: eager, cuda_graph, model_fused, model_deep, model_deep_tuned.
  These are **not** claimed to recreate historical cumulative C1-C5 toggles exactly.
  `PF_MODES=eager,cuda_graph` or `PF_STEPS=10` narrows the original sweep.
- Current Joint then runs its existing native N1/N4/N10 and parallel N4/cache matrix,
  using an independent process on the same visible GPU, followed by a separate profiler.
- `ALLOW_PARITY_FAILURE=1` is optional and affects only Joint's finite stage-partition
  diagnostic gate. A failed gate remains visibly failed; cached trajectory parity is fatal.
- Results use a fresh `COMPARE_ROOT` (default timestamped outputs directory). Original
  per-mode logs, options and official receipts are retained. The top-level `comparison.md`
  separates panels and computes eager-relative ratios only within ParaFlow at the same N.

## Fairness boundary

This benchmark fairly reports **within-workload optimization effects**, not a same-model
head-to-head algorithm comparison. Original ParaFlow uses its isolated checkpoint and
first-frame-only semantics, and raw-camera-to-action boundary. Current Joint uses a mutual
checkpoint, future-video generation and composed-canvas replay. They also consume different
observations and use different warm/cold selection protocols. Do not subtract their module
times or interpret a cross-panel ratio as an algorithm speedup. Original modes execute in
separate processes, not paired/interleaved, so repeat independent runs for robust estimates.

Only currently implemented Joint optimizations are measured: schedule caching. Conditioning
K/V reuse, prefix prefill, compilation and graphs are not silently enabled for Joint.
Neither panel measures closed-loop quality; historical latency reproduction is not asserted.
A strict same-workload comparison still requires a validated original mutual backend or an
independently validated first-frame port; this runner does not pretend either exists.
