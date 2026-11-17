# OpenWAM Joint：视频主导的自适应并行去噪实验

## 1. 研究边界与当前状态

保留 mutual Joint checkpoint、视频—动作混合注意力、N10 离散网格及首帧条件。主要研究**同一次请求内的视频去噪步骤并行**；动作仍参与联合前向和原有 Euler 更新，只提供可关闭的安全检查，不设计动作专用加速算法。

本次新增独立实验工程，不修改旧模型、部署源码、RoboTwin 源码或旧实验脚本。旧实验的源码身份不因本次新增模块改变。没有运行 GPU 测速、视频质量实验或仿真闭环，也没有新的成功率与加速结论。

与已有实现的区别：

| 方法 | 初始化与依赖 | 计算策略 |
|---|---|---|
| Native N10 | 同请求串行状态 | 10 次依次前向 |
| legacy_n10 | 现有跨请求历史及 phase 对齐 | 历史轨迹阶段批量更新 |
| fixed_k*_r* | 同请求锚点复制，不复用历史 | 固定窗口、固定次数 Picard 修正 |
| residual_k* | 同请求锚点复制 | 视频残差控制修正和连续前缀接受 |
| adaptive | 同请求锚点复制 | 视频残差同时控制窗口与修正 |

legacy 的历史初始化、噪声演化和新算法不同，最终输出差异包含这些算法差异，不解释为同输入算子的误差。

## 2. 已实现的求解器

入口在 `benchmarks/video_parallel/solver.py`，通过原有 `joint_sampler.sample` 协议注入，不改模型权重和默认行为。

- 并行窗口 K=1/2/4；联合计算窗口中多个去噪时间点。
- 从已接受锚点做 Euler 前缀积分，不把独立视频批处理当作时间并行。
- 每轮固定首帧，inactive action 维度沿解析噪声路径更新。
- 视频误差只统计未来 latent：归一化全局 RMS 与空间分块误差 P95。
- 动作检查只统计有效维度；视频与动作阈值分别设置。
- residual/adaptive 最多修正 K 轮，只接受连续前缀；无可接受前缀时从原锚点串行重算该窗口。
- 非有限批量预测触发串行重算；串行仍产生非有限值则立即失败。
- adaptive 从 K=2 开始，两次连续一次修正全部接受后升到 K=4；回退后执行一次 K=1，再恢复尝试 K=2。
- 固定预算基线可以输出未收敛结果，必须通过质量评测，不能以其低延迟证明有效。

默认参数只是可扫描的研究起点，不是已校准配置：

```json
{"policy":"adaptive","window":4,"rounds":1,"video_tolerance":0.003,"local_tolerance":0.01,"action_tolerance":0.003,"action_guard":true,"local_guard":true,"block_size":4}
```

`rounds` 只限制 fixed 策略；residual/adaptive 的最大修正次数等于当前窗口大小。初始化、残差计算和回退均计入请求时延。残差通过不构成 Native 等价性证明。

## 3. 先运行的指令

所有指令在仓库根目录执行。脚本默认使用 `.venv-openwam/bin/python`。

### 3.1 无需 GPU 的资产检查

历史 trace 没有 hash 绑定的 checkpoint sidecar，所以必须显式允许作为诊断输入：

```bash
ALLOW_UNVERIFIED_TRACE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/profile_video_parallel.sh preflight
```

此步骤只检查路径、mutual 配置、观测形状、存储空间和源码身份；`gpu_validation=not_run`。有合法 sidecar 时验证 trace/config 哈希，若提供权重路径、大小和时间戳，也会逐项核对。

### 3.2 第一个短验证实验

先选择一张可用的独占 GPU。正式入口发现其他计算进程会拒绝测速，不会停止其他作业。

```bash
GPU=0 WARMUP=2 MEASURE=4 VALIDATION_SAMPLES=1 \
METHODS=native_n10,fixed_k2_r1,residual_k2,adaptive \
ALLOW_UNVERIFIED_TRACE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/profile_video_parallel.sh profile
```

默认会检查 K1 与 Native、阶段批量与逐项前向。若批量 parity 失败，先查看 `validation.json`。只有需要继续**诊断性测速**时才加 `ALLOW_PARITY_FAILURE=1`，阈值仍为 `rtol=atol=0.05`。K1 检查失败或输出非有限值不能被此开关绕过。

### 3.3 主回放对照与阈值扫描

```bash
GPU=0 WARMUP=10 MEASURE=30 ALLOW_UNVERIFIED_TRACE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/profile_video_parallel.sh evaluate
```

默认包含 Native N10/N4/N6/N8、legacy N10、K2/K4 一次修正、固定窗口残差停止及 adaptive。

```bash
GPU=0 METHODS=native_n10,residual_k2,residual_k4,adaptive \
ALLOW_UNVERIFIED_TRACE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/profile_video_parallel.sh sweep --tolerances 0.001,0.003,0.01
```

阈值扫描将空间阈值设为视频 RMS 阈值的三倍，动作阈值保持不变。验证集选择后，将完整配置固定为 `SOLVER_OVERRIDES`，测试集不重新调参。

```bash
SOLVER_OVERRIDES='{"action_guard":false}' ... bash scripts/profile_video_parallel.sh evaluate
SOLVER_OVERRIDES='{"local_guard":false}' ... bash scripts/profile_video_parallel.sh evaluate
```

上面的 `...` 表示补入前述 GPU、TRACE 等环境变量，不是可直接执行的命令。

### 3.4 包含 VAE 解码与视频样本

```bash
GPU=0 DECODE_VIDEO=1 METHODS=native_n10,native_n6,adaptive \
ALLOW_UNVERIFIED_TRACE=1 \
TRACE=/home/zhijian/OpenWAM-main/outputs/joint_profile_20261029_231836/observations.npz \
bash scripts/profile_video_parallel.sh evaluate --video-samples 3
```

默认不保存视频；本指令只保存计时开始后的 3 个配对样本，排除固定首帧，生成 `quality_manifest.json` 与 `video_quality.json`。SSIM、PSNR 和时间差分误差在推理计时之外计算。相同视频的 PSNR 记为 null，并用 `identical=true` 标识，避免写入非标准 Infinity。

## 4. 闭环质量测试

独立 Python 代理只替换评测器启动的 `scripts/deploy.py`，其他 Python 调用原样转发。模型仍通过官方部署 CLI 加载；编译、DiT cache 和 CFG 必须关闭。新增采样器只由显式实验入口启用。

### LIBERO 筛选：8 个任务，每任务 5 回合

```bash
GPU=0 METHODS=native_n10,native_n6,legacy_n10,adaptive \
bash scripts/evaluate_video_parallel.sh
```

为不改动已有 pinned 协议，筛选采用四个套件各 task 0/1、每任务 5 回合，**不是原构思的每任务 10 回合**。正式评测仍为 40 个任务×50 回合：

```bash
GPUS=0,1,2,3 STAGE=full METHODS=native_n10,native_n6,legacy_n10,adaptive \
bash scripts/evaluate_video_parallel.sh
```

`native_n6` 是待筛选基线，不预先声称它是最佳质量匹配方案；应根据留出验证集替换为 N4/N6/N8 中合适者。

各方法共享相同任务、种子和协议，结果各自保留。LIBERO 自动生成 `quality_comparison.json/md`，按共同完成任务计算成功率差异及任务簇 bootstrap 区间；未完成不计失败，也不由筛选集宣称非劣。

### RoboTwin 迁移：10 个预先固定任务×10 回合

```bash
BENCHMARK=robotwin GPU=0 METHODS=native_n10,native_n6,legacy_n10,adaptive \
bash scripts/evaluate_video_parallel.sh
```

自动读取已有 `outputs/robotwin/environment.sh`。可用 `TASKS=adjust_bottle,open_laptop,stack_blocks_two` 提前限定任务；不要根据结果筛选任务。默认使用任务表前 10 项，`STAGE=full` 仍为所选任务、每任务 50 回合，并非自动变成全 50 任务。沿用原评测器的 sealed manifests、断点续跑、进程所有权和结果汇总。

闭环可与其他进程共享 GPU，但时延不作为正式性能证据。四卡仅用于任务分片，不是单请求四卡加速。

### 等待空闲 GPU

两个新实验脚本均支持 `WAIT_FOR_GPU=1`（默认关闭）：

```bash
WAIT_FOR_GPU=1 GPU=auto TRACE=/实际/observations.npz bash scripts/profile_video_parallel.sh profile
WAIT_FOR_GPU=1 GPU=auto bash scripts/evaluate_video_parallel.sh
WAIT_FOR_GPU=1 GPUS=0,1 bash scripts/evaluate_video_parallel.sh
```

`auto` 自动选择一张物理卡，指定多卡则等待所有指定卡同时空闲。
默认每 30 秒检查，连续空闲 60 秒后启动；空闲条件为无 compute 进程、
显存使用不超过 1024 MiB、利用率不超过 5%。通过 `GPU_POLL_SECONDS`、
`GPU_IDLE_SECONDS`、`GPU_MAX_MEMORY_MIB`、`GPU_MAX_UTILIZATION` 调整。
`GPU_WAIT_TIMEOUT=0` 表示无限等待，非零为等待超时秒数。Ctrl-C 取消等待。
查询失败直接报错，不视为空闲；`preflight`、`report`、闭环 `--summary` 不等待。

启用时请取消 `CUDA_VISIBLE_DEVICES`，使用物理 `GPU`/`GPUS` 环境变量，
不要另传覆盖设备的 `--device`/`--gpus` 参数。不杀进程、不提前占显存。
同用户的等待器使用按 GPU UUID 的协作文件锁，实验主进程退出后释放。
其他用户或外部程序不受该锁约束，不能保证机器级独占；原有性能独占检查保留。

后台等待示例（需替换 TRACE）：

```bash
nohup env WAIT_FOR_GPU=1 GPU=auto TRACE=/实际/observations.npz \
  bash scripts/profile_video_parallel.sh profile > /tmp/video_parallel_wait.log 2>&1 &
```

### tmux 通宵串行队列

在可访问 GPU 的主机终端执行（默认使用已有历史 trace，自动等任意一张卡）：

```bash
cd /home/zhijian/OpenWAM-main
tmux new-session -d -s video-night 'bash scripts/overnight_video_parallel.sh; exec bash'
tmux attach -t video-night
```

`Ctrl-b` 后按 `d` 脱离，不停止队列。禁止重复启动相同会话；再次进入使用 attach。
需要指定卡或输出目录时，在 tmux 命令内部加入环境变量，例如
`NIGHT_GPU=2 QUEUE_OUT=outputs/my_night bash scripts/overnight_video_parallel.sh`。
通宵队列默认 `NIGHT_GPU=auto`，忽略终端继承的 `GPU`（例如旧任务留下的 `GPU=0`），
避免明明有其他空闲卡却始终等待 0 卡。单项启动脚本仍使用 `GPU` 设置。
队列检查点默认固定为 `/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO`，不继承旧任务的
`CKPT_DIR`/`CKPT_NAME`；需要替换时设置 `NIGHT_CKPT_DIR`/`NIGHT_CKPT_NAME` 并提供匹配的 `TRACE`。
实际检查点路径写入队列日志；等卡前先运行 CPU preflight，结果保存在 `preflight.log`。
每次任务会重新等待 GPU，不杀其他进程，也不同时占满所有卡。
队列包括 smoke、10 方法主比较、3 阈值扫描、固定迭代预算、移除局部残差保护、
两个额外随机种子、包含视频解码计时，共 8 项；耗时由硬件和等待决定，不保证跑满或跑完一夜。
默认正式计时预热 10、测量 60 请求，可用 `NIGHT_WARMUP`/`NIGHT_MEASURE` 修改。
默认不保存像素视频，保留 5 GiB 磁盘检查；磁盘不足时不会继续加载模型。
可显式设置 `MIN_FREE_GIB=3` 将输出磁盘余量门槛改为 3 GiB（默认仍为 5），
这只是检查门槛，不是空间预留或磁盘使用上限；优先清理确认无用的数据或改用空间充足的输出盘。
结果目录包含 `queue.log`、`status.tsv` 和各项独立输出。Smoke 失败立即停止，
后续单项失败记录后继续，所有失败都需检查，不能当完成实验。
默认不绕过批量数值一致性检查；已知历史 trace 以 unverified diagnostic 标记，
不自动开展耗时闭环，也不根据测试结果选择阈值或修改质量门槛。

## 5. 产物与解释

离线结果默认位于 `outputs/video_parallel_时间戳/`：

| 文件 | 内容 |
|---|---|
| identity.json | checkpoint/trace/源码身份、实验配置、硬件快照 |
| validation.json | K1、批量分区及近似输出诊断 |
| requests.jsonl | 每个请求的时延、显存、窗口、修正、回退及输出误差 |
| profile.json | 独立 instrumented pass 的区域耗时与调用次数 |
| comparison.json/md | 完成状态、诊断标签、P50/P95 与误差 |
| curves.csv | 延迟—误差绘图数据 |
| failure.json | 失败原因，已有结果保留 |

默认计时包括条件准备、完整 Joint 视频 latent 去噪和 CPU 动作输出；`DECODE_VIDEO=1` 才包含像素解码。不能把跳过解码标成完整视频生成。当前 `solver_host_ms` 是求解器主机计时，不当作独立精确 GPU 核时间。

profiler 中 Joint 前向是 inclusive 区域，与视频、动作和混合注意力重叠，不能相加。未捕获的区域不视为零。显存总峰值包含共享引擎及比较采样器保留状态，逐请求另有增量峰值；正式容量结论需要每方法独立进程复测。

恢复报告：

```bash
OUT=/实际/实验目录 bash scripts/profile_video_parallel.sh report
.venv-openwam/bin/python -m benchmarks.video_parallel.closed_loop \
  --output /实际/LIBERO闭环目录 --summary
```

### 独立视频参考集

`python -m benchmarks.video_parallel.quality --manifest manifest.json --output quality.json` 可评估外部配对 NPZ。manifest 格式：

```json
{"samples":[{"method":"adaptive","scene":"heldout_scene_01","sample_id":"0","reference_kind":"matched_real_future","conditioning_actions_equal":true,"path":"pair_000.npz"}]}
```

NPZ 包含 `candidate_rgb` 与 `reference_rgb`，均为相同形状的 uint8 `[T,H,W,3]`。真实未来必须确认动作条件相同；与 Native 的保真度使用 `reference_kind=native_fidelity`，两类指标分开汇总。输入相同条件的真实性由采集协议保证，不能仅凭布尔字段当作自动认证。

当前不自动构造动作固定的模型路径，不把 Joint 自生成动作对应的视频与不匹配的真实轨迹比较。LPIPS、物体运动识别和人工盲评标记为未运行，需后续固定工具/数据后补充，不用 SSIM 冒充这些指标。

## 6. 继续/停止与论文目标

1. 先跑短实验，确认 K1 正确、数值误差可解释，再观察 K2/K4 的真实批量耗时与回退成本。
2. 若不存在质量可接受且去噪至少降 20%、完整请求至少降 10% 的点，停止扩展调度，不先训练新网络。
3. 若存在空间，扫描阈值并比较少步串行，冻结验证集选择后再开展完整视频与闭环评测。
4. 完整请求 1.3×、闭环下降不超过 2 个百分点是研究目标，不是当前结果。只有任务聚类统计和质量证据支持，才宣称非劣。
5. 当前版本未接入跨请求历史初始化、ParaTAA 加速求解器、LoRA 或学习型调度；这些保留为后续、有独立对照的扩展，不能声称已复现。

CPU 验证：

```bash
.venv-openwam/bin/python -m unittest \
  tests.benchmarks.test_video_parallel \
  tests.benchmarks.test_joint_stage_report \
  tests.benchmarks.test_joint_repeatability
```

第一版已具备从资产检查、算法验证、离线扫描到复用现有闭环评测器的执行链路。是否具有论文级收益，必须由上述后续实验决定。
