# H2 one-night robustness study

This is a diagnostic follow-up to
`experiment/e2_exploratory_20261113_205159_787207`, not a deployment benchmark.
It does not run held-out tasks, claim success-rate preservation, or implement
inference acceleration. No experiment starts merely by installing these scripts.

## Launch

From the repository, run:

```bash
mkdir -p experiment && (nohup env GPU=4 bash scripts/research_alpha_libero_e2_robustness.sh > "experiment/e2_robustness_launch_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null & echo "Launcher PID=$!")
```

`GPU` is the physical `nvidia-smi` index. The launcher resolves and logs its UUID;
the process uses logical `cuda:0`. It never switches GPU automatically. The log
prints the unique run directory, PID, UUID, check completion, first completed pair,
and every 100 newly executed pairs with an updated ETA. A launcher PID alone is
not proof that the model has loaded or that experiments are progressing.

Default budget is ten hours per invocation, including preparation/model load.
The soft limit stops dispatching new work at a safe pair/check boundary; model
loading, an active generation, and final CPU reporting are not forcibly killed.
Historical throughput suggests about 8–12 hours for the full workload, not a
guarantee. No other user's GPU process is stopped.

Resume after a stop using the printed run directory, with the same GPU, budget,
checkpoint and source files:

```bash
RUN=/absolute/path/to/experiment/e2_robustness_TIMESTAMP_PID GPU=4 bash scripts/research_alpha_libero_e2_robustness.sh
```

Each resumed invocation has a new ten-hour allowance and reruns fresh correctness
checks. Completed, sealed, identity-matching pairs are skipped. Failed attempts
are preserved separately; unexpected identity changes or corrupt results stop the
run. Concurrent processes cannot operate on the same run directory.

## Frozen protocol

- Reuse the original 24 states per development split, layer/token layout, policy
  seed 43 and Calibration floor. Original files remain untouched and are hashed.
- Stage A: both splits, layers 0/29, steps 1/5/9, eight groups, epsilon 0.03,
  new direction seed 20261114, paired positive/negative perturbations: 4,608 pairs.
- Stage B: both splits, layers 0/29, step 9, eight groups, epsilons 0.01/0.10,
  original direction seed 20261112, paired signs: 3,072 pairs.
- Compare to the original complete 4,608 pairs at epsilon 0.03. New workload is
  7,680 pairs, approximately 15,360 full generations plus correctness checks.
- Execute A Discovery, A Calibration, B Discovery, B Calibration. A partial run
  remains partial; no unobserved condition is treated as passing.

The model/source identity and original T0 certificates must match. New scripts
live outside the model identity's source scan, so no source-review whitelist is
expanded. The standalone tool itself and the original input hashes are sealed
in the new plan. Do not edit model or research sources to bypass these checks.

Each fresh process checks fixed selected states at offsets 0, 8 and 16 in each
split (repeat baseline, gate=1 and saved baseline). For each selected layer/step,
it additionally measures gate=1 and diagnostic-repeat prefix normalized RMSE.
These are like-for-like metric floors, not ratios to native maximum error.
The old T0 native repeat maxima are also reported. Numerical floors are measured
on these control states, not every experimental state; zero floor does not rule
out precision/quantization effects.

## Reports and decisions

`status.json` reports `complete`, `budget_exhausted` or `failed`.
`progress.json` is an execution-time progress snapshot; `report.json` contains
the authoritative completed count after reporting. `results/` stores sealed
pairs; `errors/` preserves failed attempts; `checks/` preserves per-process checks.
`report.md` and `report.json` are regenerated even after an execution error.

The report keeps direction, amplitude, layer, step and split separate. It reports
visual-change/sensitivity correlations, cross-condition ranking Spearman and
Top-2 overlap, zero/tied responses, magnitude sign asymmetry, and the fraction
above 3 times the measured floor (null when floor is zero). Bootstrap first
averages states within episode, resamples episodes within task and tasks, then
uses task-uniform means: 10,000 replicates, seed 20261114. These small development
samples give descriptive intervals, not confirmation on held-out tasks.

Calibration ranking eligibility requires all 24 states/eight tasks, mean
Spearman >= 0.5 and mean Top-2 overlap >= 0.5. Step 9 must also pass both amplitude
comparisons; other steps are explicitly not amplitude-validated. Constant
rankings have undefined correlation and cannot pass. The report includes raw
state and task summaries. Passing only qualifies a position for profiling, not
safe deletion or deployment. If rankings fail, pause H2 group selection and
return to H1 combination experiments; if responses are quantization-dominated,
improve measurement first. Binary gripper-flip claims are not made from the old
numeric `gripper_disagreement` metric.

## CPU-only preparation and tests

Use a separate temporary run directory for preparation (it has no GPU identity
and is not a GPU-resume directory):

```bash
PREPARE_ONLY=1 RUN=/tmp/e2_robustness_prepare bash scripts/research_alpha_libero_e2_robustness.sh
.venv-openwam/bin/python -m pytest -q scripts/inference_test/test_e2_robustness.py
```

Rebuild a report without loading a model:

```bash
PYTHONPATH=. .venv-openwam/bin/python scripts/research_e2_robustness.py --run /absolute/run/directory --report-only
```

Direct Python options also support `--base`, `--bank-run`, `--split`, `--ckpt-dir`,
`--ckpt-name`, and `--hours`. The wrapper forwards options; `HOURS=10` sets its
default budget. Changing any frozen input requires a new run directory.
