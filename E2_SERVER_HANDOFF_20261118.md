# E2 robustness 迁移交接（2026-11-18）

## 当前结论

本服务器任务已按用户要求停止。已向主进程 50794、模型子进程 51240 和扩展监督器 54242
发送 SIGTERM，并确认均退出；无本次实验遗留 worker。全部结果和日志保留。
运行目录内 `parallel_status.json`、`extensions/status.json` 已标记 `stopped_by_user`，
停止前状态归档在 `stop_history/`。没有完成 robustness，也没有启动新增干预配对。

## 目标和已确认的偏好

- 在迁移后的仓库复现原 `scripts/research_alpha_libero_e2_robustness.sh`。
- 用户选择在新硬件上独立复现：重新采集状态、T0、特征、校准及 E2 基线，保留身份校验。
- 优先寻找和复用服务器本地模型文件；避免传输 24 GB 参数。
- 使用多张显卡分配独立状态/配对；不改变模型、bf16 精度、10 步采样或计算定义。
- 新增实验优先为干预对照及多种子验证。后续另加不同 episode 和扰动扫描。
- 最后一次约定预算为从 11 月 17 日约 23:19 起十小时，至 11 月 18 日 09:19（北京时间）；
  现已主动停止。迁移后需重新明确新的时间预算，不能沿用过期的绝对截止时间。

## 产物清单

本仓库绝对路径：`/mnt/storage/users/zjfang_data/OpenWAM-private`。

新实验目录：`experiment/e2_reproduction_20261117_215825_3789091/`（约 159 MB）。
原服务器参照：`experiment/reference/experiment/`（约 165 MB）。

| 阶段 | 有效产物/进度 |
|---|---|
| 模型资产核验 | checkpoint、配置、归一化统计及身份涉及的 tokenizer 文件匹配原记录 |
| EGL 与模型推理预检 | 通过；动作形状 32×10，实测峰值分配约 23.4 GiB |
| Discovery 新状态库 | 214 个状态，8 个开发任务，每任务采集 10 个 episode |
| Calibration 新状态库 | 218 个状态，同样的采集协议 |
| T0 | 两个 split 各 20 个状态，种子 43/44/45，均通过 |
| 跨 GPU T0 | 原 GPU 3、6 对 Discovery 的额外验证均通过 |
| Representative layouts | 固定层 0、29，已在新 Discovery 上取得布局 |
| 冻结选择 | 每个 split 24 个状态，8 个任务×3 个状态；层 0/29，步 1/5/9 |
| 特征及 Calibration | 两个 split 均完成；Calibration floor 已冻结 |
| Discovery E2 | 完成 2304/2304 对，全部 completed，已合并并重新校验 |
| Calibration E2 | 计划已生成；smoke 未完成，无完整 E2 配对结果 |
| Robustness | 未开始，无 `robustness/` 实验结果目录 |
| 新增扩展 | 协议已冻结并入队，但因主实验失败未开始；无 `extensions/plan.json` 和配对结果 |

关键相对路径：

```text
bank/state_bank/{discovery,calibration}/        # 状态、原生动作、模型/环境/manifest
base/diagnostics/t0_{discovery,calibration}/    # T0 报告与 sealed 证书
base/diagnostics/features_{discovery,calibration}/
base/manifests/selection.json
base/manifests/e2_{discovery,calibration}.json
base/diagnostics/e2_discovery/                 # 2304 条结果 + manifest/plan
parallel/replay_discovery/worker_{0,1,2,3}/    # 保留原分片供追溯
parallel/cross_gpu_t0/                        # 跨卡 T0
parallel/logs/                               # 各 worker 日志，包括失败与重启输出
extensions/protocol.json                      # 新增实验预定方案
extensions/protocol_history/                  # 预算/GPU 更新前的协议
parallel/recovery.json                        # 已执行的受校验恢复记录
stop_record.json                              # 停止记录
```

## 模型和环境

本机本地权重来源：

```text
/mnt/storage2/users/hycui_data/yyf/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO
```

完整模型参数没有从原服务器传输。仓库中的 `.cache/models/OpenWAM-Alpha-Sim-LIBERO/`
是指向该目录的软链接组合，仅补取了缺少的 `download-manifest.json` 元数据。
新服务器应寻找自己的完整本地模型目录，重新建立路径，避免直接迁移这些绝对软链接。

Checkpoint：`checkpoint_step_10690.safetensors`，大小 24,813,767,464 bytes。

```text
checkpoint SHA-256:
34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a

本次模型身份 SHA-256:
38025c76e5f9ed690ebd9591af19b026dd5bb7dd02bf536770fa6b78dd7b626c

研究源码 SHA-256:
63318e1592c24eb10dbe4ceef306cc69b95ba9bb613b606a02f63fb9c5a02b77
```

本次环境记录：Python 3.10.21，PyTorch 2.7.1+cu128，CUDA runtime 12.8，
NVIDIA GeForce RTX 4090（本机显存约 48 GiB），Linux 6.8.0-146-generic / glibc 2.39。
注意力环境变量 `WAM_ATTENTION_IMPL` 与 `DIFFSYNTH_ATTENTION_IMPLEMENTATION` 均未设置。
完整模型身份与运行包版本保存在 `bank/state_bank/*/model.json` 和顶层 `environment.json`。

`.venv-openwam`（约 953 MB）通过内部 `.pth` 只读引用本机已有 OpenWAM 环境，
它不是可移植环境。新服务器必须重建依赖。原服务器 pip 清单在
`.cache/migration/source-pip-freeze.txt`（可能包含旧 editable 安装路径，不应直接原样安装）；
本次实际安装包记录以 `environment.json` 为准。

本机依赖设置：research requirements 固定了 robosuite 1.4.0、MuJoCo 3.3.2 等版本。
本地 robosuite `macros_private.py` 关闭共享 `/tmp/robosuite.log` 文件日志，日志由启动器收集。
LIBERO 使用仓库内 `third_party/LIBERO`，EGL 环境变量和缓存目录由启动器设置。

## 中止和失败原因

原 A100 服务器的旧 robustness 在 0/7680 时已失败，原因是 model/backend identity 不一致。
因此本次独立建立了新的 4090 状态库与基线，没有将旧结果伪装为新实验结果。

本服务器共享 GPU：启动前空闲显存检查不能阻止其他用户之后分配显存。
11 月 18 日约 00:13，Calibration smoke 在 GPU 0 加载模型时，另一进程占用了约 30 GB，
本进程只有约 17 GB 余量，触发 CUDA OOM。编排器按失败停止，扩展监督器随后停止。

恢复修改包括：取得模型加载锁后再次检查至少 28000 MiB 余量；仅针对模型加载阶段的
CUDA OOM 清理后等待重试；数值、配置和推理失败仍停止。新 worker 每个默认 4 CPU 线程，
模型加载峰值用文件锁串行，实际推理可并行。

约 00:24 已从失败处受校验恢复，改用 GPU 2/3/4/6，保留全部完成结果；
随后用户要求迁往另一服务器，恢复中的模型加载已停止。当前不再有后台任务。

## 代码改动（尚未提交）

- `scripts/reproduce_alpha_libero_e2_robustness.sh`：本地权重、环境、日志入口。
- `scripts/reproduce_e2_robustness.py`：单卡完整独立复现编排。
- `scripts/reproduce_e2_parallel.py`：多卡采集接管、分片、合并、受校验 Calibration smoke 恢复。
- `scripts/research_parallel_worker.py`：独立 GPU worker、加载锁与显存处理。
- `scripts/research_e2_robustness.py`：保留原校验，增加按状态分片和 GPU 分配记录。
- `scripts/research_e2_extensions.py`：有截止时间的预定扩展实验、监督器及分析。
- `scripts/inference_test/test_{reproduce_e2_robustness,e2_parallel,e2_extensions}.py`：新测试。
- `MIGRATION.md`：启动方式、环境与限制。

测试记录：原研究相关验证曾通过 91 项；后续分片/扩展组合通过 30 项，
最后显存检查/恢复相关套件通过 9 项。EGL、真实模型推理、新状态采集和 Discovery E2
均实际执行过。Robustness 与扩展 worker 尚未完成端到端 GPU 验收。

只做 `git clone` 不会得到这些未提交改动；需复制改动或使用随附迁移包。

## 新服务器建议执行顺序

1. 恢复同一仓库快照及上述未提交代码；带走本次运行目录和原参照目录。
   不传输虚拟环境、密码、SSH socket 或本机权重软链接。
2. 优先寻找本地完整 LIBERO checkpoint，核验上述 SHA-256，并核验所有 model asset 清单；
   缺少 `download-manifest.json` 时从本次缓存/迁移包补齐元数据。
3. 重建依赖，运行相关 CPU 测试、EGL 及单卡真实模型推理。
   建议使用独占/预约 GPU，并在共享机器上保留 28 GB 以上空闲显存和足够主机内存。
4. 先判断是否允许复用本次模型身份。身份包含 GPU 型号、Python/PyTorch、平台内核等，
   新服务器任一变化都可能导致严格校验失败。若不一致，建立新 RUN 重新采集并生成基线；
   不改写旧 sealed manifest，不放宽身份校验。可以复用原任务划分和实验协议作独立复现。
5. 如果整套身份确实完全相同，可基于现有结果实现接续 Calibration；注意本机
   `--resume-verified` 专门处理 `smoke_calibration` 失败状态，当前状态已被主动停止，
   不能在新机直接无条件调用。应增加显式的停止后恢复分支、归档停止记录并重新验证，
   而不是把 `stopped_by_user` 手改成 `failed`。
6. 完成 Calibration 的 2304 对后，运行原 Robustness 的 7680 对，并生成报告。
7. 接续预定扩展：按 Discovery 等任务/episode 权重拟合敏感性排名；2/4/6 个分组预算下，
   比较低敏感、高敏感、均匀及五个随机对照；分别 gate=0 和 stale-V；用策略种子 44/45
   重复验证，之后用不同 saved episode ID 和额外噪声方向/幅度补充。

扩展完整队列容量为 96768 对，按固定优先级运行至新截止时间，不保证十小时全做完。
主指标为 Calibration 上低敏感对照相对随机对照的动作 RMSE 差，只有完整匹配策略组
才进入比较，并提供 episode/task bootstrap 区间。所有这些仍是开发任务上的固定观测
机制实验，不是闭环成功率、未见任务泛化、真实 token pruning/cache 或速度提升的证明。

启动完整新实验的一般形式（需先配置可用的本地模型目录）：

```bash
mkdir -p experiment
setsid nohup env GPU=auto HOURS=10 CKPT_DIR=/NEW/LOCAL/OpenWAM-Alpha-Sim-LIBERO \
  bash scripts/reproduce_alpha_libero_e2_robustness.sh \
  > experiment/new_server_launch.log 2>&1 < /dev/null &
echo "PID=$!"
```

这里 `HOURS` 仅是最终 robustness 阶段预算；如需总墙钟预算，必须配置新的统一 deadline。
多卡编排目前面向采集阶段接管，须用新机实际 PID/GPU 重建接管命令，不能复制本机旧 PID。

## 原服务器参照

原资产仍在 `zhijian@112.30.139.26:60028` 的 `/home/zhijian/OpenWAM-main/experiment/`，
权重目录为 `/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO`。认证信息不写入本交接文档。
本次已取回所需小体积参照；新服务器优先使用迁移包，通常不必再连接原服务器。
