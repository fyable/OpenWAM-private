# Joint chunk pipeline for LIBERO

`chunk-pipeline` uses the Alpha checkpoint with mutual Joint attention, 10 native Euler updates, a window of 32 actions, and 8 executed actions per request. Each chunk keeps one action and video state at its current stage. Four resident workers rotate ownership of the next request. Only the newly added farthest chunk draws initial noise after startup. The clean video prefix is re-encoded from the matching predicted frame at every background interval and replaced by the observed frame when the chunk becomes executable. Forecast proprioception holds the last real value.

With three speculative steps per interval, a fully overlapped run uses 10, 7, 4, and then one online step. The ablation with two speculative steps uses 10, 8, 6, and then four. When the next observation arrives early, the next worker finishes its current step and uses the real condition for every remaining step. A worker waiting for a prediction from another GPU can yield to real observations before the forecast arrives.

For fixed-observation replay, use an NPZ in the format described in [JOINT_PARAFLOW.md](JOINT_PARAFLOW.md). Set `TRACE` to the observation path and `EXEC_MS` to the measured time budget for executing eight actions. Run the same trace on the native baseline and both pipeline placements.

```bash
export E2_MEMORY_MODE=text-offload
CKPT=.cache/models/OpenWAM-Alpha-Sim-LIBERO
PY=.venv-openwam/bin/python
COMMON=(--ckpt-dir "$CKPT" --observations "$TRACE" --execution-ms "$EXEC_MS")
"$PY" -m benchmarks.libero.replay_chunk_pipeline "${COMMON[@]}" \
  --sampler native --output experiment/chunk_replay/native.json
"$PY" -m benchmarks.libero.replay_chunk_pipeline "${COMMON[@]}" \
  --sampler chunk-pipeline --output experiment/chunk_replay/one_gpu.json \
  --baseline experiment/chunk_replay/native.npz
"$PY" -m benchmarks.libero.replay_chunk_pipeline "${COMMON[@]}" \
  --sampler chunk-pipeline --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --output experiment/chunk_replay/four_gpu.json --baseline experiment/chunk_replay/native.npz
"$PY" -m benchmarks.libero.replay_chunk_pipeline "${COMMON[@]}" \
  --sampler chunk-pipeline --devices cuda:0,cuda:1,cuda:2,cuda:3 --speculative-steps 2 \
  --output experiment/chunk_replay/four_gpu_6plus4.json --baseline experiment/chunk_replay/native.npz
```

To run a closed loop, use the regular LIBERO launcher with `--sampler chunk-pipeline`, `--inference-horizon 8`, and `--replicas-per-gpu 1`. `--gpus 0,1,2,3` creates one server that owns all four devices for one rollout at a time. `--gpus 0` uses the same pipeline on one device. The server log records worker placement, stage at observation arrival, online steps, action-ready latency, background wait, decode duration, and peak per-device GPU memory. Compare rollout success and elapsed time with `--sampler native` on the same tasks, trial indices, seed, and E8 settings.
