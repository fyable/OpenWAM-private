# OpenWAM 四卡 chunk 流水线复现（2026-11-20）

## 已跑的策略

使用 OpenWAM Alpha Sim LIBERO 的 Wan2.2 TI2V-5B Joint 模型，保留双向视频与动作注意力。采样为 **N10/H32/E8**，即视频和动作一起做 10 次去噪，每个 chunk 预测 32 个动作，只执行前 8 个。视频对应 33 个原始时间位置，按 stride 4 取 9 帧。每个 chunk 在一张卡上维护当前去噪节点，包括视频 latent、动作 latent 和步数。四张卡各加载一份模型，轮流负责头部 chunk。

1. 首个请求用真实观测完成 10 步。头部 chunk 动作就绪后，后台解码它的视频，提取相对于该 chunk 起点的 `+8/+16/+24` 预测帧，对应解码帧索引 `2/4/6`。
2. 环境执行 E8 的同时，其余三个 worker 分别用各自对应的预测帧推进后续 chunk。每轮各最多推进 3 步，已有 chunk 沿用上次节点，新进入最远位置的 chunk 从噪声开始。每轮使用上一头部 chunk 新生成的预测帧作为条件。后台累计最多 9 步。
3. 新真实观测到达后，停止新头部 worker 的后台推进并等待在途工作返回，编码真实首帧并替换视频首帧 latent，沿用其它视频 latent、动作 latent 和当前步数，做完剩余 `10 - stage` 步。执行前 8 个动作后窗口继续前移。每个 episode 重置流水线。

`optimization.decode_video=false` 关闭部署接口的视频输出。流水线仍在后台解码预测帧并运行视频去噪。当前实现的预测帧来自 Joint 模型的预测，并未在真实环境提前执行对应动作。节点 9 是后台上限。已跑长测的稳态在线补步主要为 3 或 4 步。

## 迁移到新服务器

从当前工作目录同步源码快照，包含 `openwam/`、`benchmarks/libero/`、`scripts/`、`configs/`、`tests/`、`third_party/LIBERO/` 和安装所需的仓库根目录文件。迁移时可以跳过已有实验输出、本机 `.venv-openwam`、`.cache` 和私有配置。务必带上当前工作目录里的流水线改动。用下列指纹核对新机器上的核心文件。

```bash
sha256sum -c <<'HASHES'
cf9f312bc189bbde41bc808317984f161d7377f2e2b5ced641215a9141377a5c  openwam/deploy/chunk_pipeline.py
00cb4643949664cdd76446c5e729afe6e279d7754196fd58deba77126a6207e0  openwam/deploy/engine.py
d4f531a6a83a923a46b7d27e78307beb37a5afc636a7e04584e1aa82264c967e  openwam/deploy/server.py
9c39ee3c768ace74821f6991b38a7f39817a493ee3231ee250cb64fea2c438b0  benchmarks/libero/scheduler.py
d4211e25b46ec5e164b3662f4ed06743bef2be9ca8243f7c41b14e631f1a89cf  benchmarks/libero/policy_config.yml
HASHES
```

原机环境记录为 Python 3.10、PyTorch 2.7.1+cu128、MuJoCo 3.3.2、robosuite 1.4.0，使用四张约 48 GiB 的 GPU 和 EGL。以下安装命令适用于支持 CUDA 12.8 的目标环境。已有可用的 OpenWAM 环境可复用。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . -r benchmarks/libero/requirements-research.txt
```

准备完整的 `OpenWAM-Alpha-Sim-LIBERO` 模型目录，包括 `config.yaml`、`normalization_stats.npy`、`tokenizer/` 和 `checkpoint_step_10690.safetensors`。源机器的模型目录含本地软链接，目标机使用自己的实际路径。checkpoint SHA-256 为 `34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a`。LIBERO 的任务描述和初始状态来自 `third_party/LIBERO/libero/libero/{bddl_files,init_files}`。

## 启动闭环对照

在目标仓库根目录执行下列脚本块。修改 `CKPT_DIR` 和 GPU 编号。四卡顺序决定 worker 到物理卡的映射，第一张卡同时用于原生对照。`BASE_PORT` 设置为空闲端口。先确认四张 GPU 有足够空闲显存。原机深度阶段四卡组曾因共享 GPU 被其它进程占用约 10 GiB 而 OOM。

```bash
set -euo pipefail
CKPT_DIR=/path/to/OpenWAM-Alpha-Sim-LIBERO
CKPT_NAME=checkpoint_step_10690.safetensors
GPU_LIST=1,2,3,0
HEAD_GPU=${GPU_LIST%%,*}
BASE_PORT=8993
RUN="$PWD/experiment/chunk_pipeline_repro_$(date +%Y%m%d_%H%M%S)"
export LIBERO_PATH="$PWD/third_party/LIBERO"
export PYTHONPATH="$PWD:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export SERVER_PYTHON="$PWD/.venv/bin/python" LIBERO_PYTHON="$PWD/.venv/bin/python"
export E2_MEMORY_MODE=text-offload REPLICAS_PER_GPU=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
"$SERVER_PYTHON" -c 'import torch, mujoco, robosuite; print(torch.__version__, torch.cuda.device_count(), mujoco.__version__)'
test -f "$CKPT_DIR/$CKPT_NAME"

common=(--denoise-steps 10 --inference-horizon 8 --inference-mode sync
  --denoise-mode sync --compile-enabled false --seed 42
  --suites spatial,goal,object,long --render-gpus "$HEAD_GPU"
  --extra-deploy-arg optimization.dit_cache.enabled=false
  --extra-deploy-arg optimization.decode_video=false
  --extra-deploy-arg inference.request_timing=true)

run_pair() {
  local block=$1
  shift
  GPUS="$HEAD_GPU" BASE_PORT="$BASE_PORT" OUTPUT_DIR="$RUN/$block/native" \
    bash benchmarks/libero/run_eval.sh "$CKPT_DIR" "$CKPT_NAME" \
      --sampler native "${common[@]}" "$@"
  GPUS="$GPU_LIST" BASE_PORT="$BASE_PORT" OUTPUT_DIR="$RUN/$block/chunk-pipeline" \
    bash benchmarks/libero/run_eval.sh "$CKPT_DIR" "$CKPT_NAME" \
      --sampler chunk-pipeline --chunk-speculative-steps 3 "${common[@]}" "$@"
}

# 每种方法 20 个回合，四套任务各取 task 0 的前 5 个初始状态。
run_pair smoke --diagnostic --task-ids 0 --num-trials 5
```

短测通过后在同一 shell 执行以下命令。覆盖阶段每种方法共 160 回合。深度阶段每种方法计划 200 回合。

```bash
run_pair coverage_a --diagnostic --task-ids 1,2,3 --num-trials 5
run_pair coverage_b --diagnostic --task-ids 4,5,6 --num-trials 5
run_pair coverage_c --diagnostic --task-ids 7,8 --num-trials 5
run_pair depth --task-ids 9 --num-trials 50
```

## 验收

- 查看各组的 `manifest.json`、`summary.json` 和 `summary.csv`。首先确认两组完成了相同任务和回合。按 suite、task id、trial 对照成功情况。四卡 manifest 的 `resume_signature` 应记录 `chunk_pipeline_gpus=4` 和 `chunk_speculative_steps=3`。
- 在 `logs/servers/` 查看 `chunk-pipeline request stats`，用 `head_worker` 验证轮转。区分 `ready_stage_at_observation` 和接管后台任务后计算的 `online_steps`。`cold=True` 表示 episode 的前三个 chunk。稳态比较用 `cold=False`，端到端对照时保持两组采用同一种延迟口径，并记录是否包含冷启动。
- 本机短测两组均为 20/20。覆盖阶段原生 156/160，四卡 157/160，观测到动作就绪延迟 P50 从 1.75 s 降到 1.21 s（1.45 倍），P95 从 2.21 s 降到 1.34 s。四卡稳态 2,711 个请求约 77% 在线补 3 步、23% 补 4 步。深度阶段仅原生组完成，结果为 199/200。

原始短测记录在 `experiment/chunk_pipeline_paired_20261120_012550_WsTJ/`，长测记录在 `experiment/chunk_pipeline_10h_E7E8/`。目录内 `run.sh` 和 `manifest.json` 可用于交叉核对。
