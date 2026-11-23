# 私有仓库与服务器迁移

## 本快照包含什么

本仓库是当前工作目录的源码快照，不包含原项目的 Git 历史。
保留源码、测试、配置、文档、许可证和版本化图片。
`third_party/LIBERO` 以普通文件保存，包含原目录的本地修改及其许可证；
导出时参考的上游提交为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，
但不代表这些本地文件与该提交完全一致。

不包含 `.venv-*`、缓存、训练输出、下载的模型、数据或本地 RoboTwin 安装。
这些内容必须在新服务器重新安装、下载或单独传输。
不要将访问令牌、私钥、环境文件或受限数据提交到仓库，即使仓库是私有的。

原目录的 Cosmos 子模块为空，且没有可恢复的原始提交号。
本快照不保留无效的子模块声明；需要 Cosmos 时按 README 克隆上游源码。
该下载不锁定版本，不能保证重现原环境。验证兼容性后，应记录实际使用的提交号。

## 首次上传

1. 在自己的 GitHub 账户下创建空的 **Private** 仓库，建议命名为
   `OpenWAM-private`。不要初始化 README、许可证或忽略文件。
2. 在终端配置 GitHub SSH 认证；不要将凭据写入仓库或发到聊天中。
3. 在本导出目录检查 `git status` 和提交清单。若尚未提交：

   ```bash
   git add .
   git diff --cached --stat
   git commit -m "Import current OpenWAM source snapshot"
   ```

4. 将以下占位地址替换为自己的私有仓库地址，然后执行：

   ```bash
   git remote add origin git@github.com:YOUR_ACCOUNT/OpenWAM-private.git
   git ls-remote origin
   git push -u origin main
   ```

   若远程已包含提交，先停止并检查，不使用强制推送。
   上传前和上传后均在 GitHub 页面确认仓库为 Private。

## 在其他服务器恢复

新服务器需要 Git、Python >= 3.10，以及适合目标 GPU 的 NVIDIA 驱动。
安装依赖时还可能需要编译器和 CUDA Toolkit，详见 README。
先为该服务器配置对此私有仓库的访问权限，再执行：

```bash
git clone git@github.com:YOUR_ACCOUNT/OpenWAM-private.git OpenWAM
cd OpenWAM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

上述 PyTorch/CUDA 组合来自现有 README，不保证适合所有服务器。
也可以按 `assets/openwam_usage_docs/docker.md` 使用 Docker 路线。
模型与基准数据通过 `scripts/download_assets/` 中的下载脚本准备；
RoboTwin 等模拟器按相应 benchmark 文档安装。
检查配置和启动脚本中的绝对路径、GPU 编号、网络端口及资产目录后再运行。
现有依赖声明并非完整环境锁定，重建环境不保证与当前机器逐版本一致。

## 验收

- 在独立目录重新克隆远程仓库，核对 `git rev-parse HEAD` 与导出仓库一致。
- 确认源码和许可证存在，虚拟环境、密钥、模型和训练输出未被追踪。
- 需要 Cosmos 时验证上游源码下载及安装；不需要时可跳过。
- 在目标 GPU 服务器验证依赖导入及选定的推理或评测任务。
- Git 推送成功仅代表代码传输完成，不代表环境、模型和 GPU 运行已验证。

## 后续同步

本导出目录与原工作目录相互独立，不会自动同步后续修改。
首次上传完成后，建议以新的 Git 仓库作为代码修改入口，并通过
`git add`、`git commit`、`git push` 同步；其他服务器使用 `git pull`。

## 在新 GPU 上独立复现 E2 robustness

### 24GB 显卡的全流程多卡入口

`scripts/reproduce_alpha_libero_e2_fresh.sh` 可直接建立独立实验，无需旧协调器 PID。
本次目标服务器有四张 RTX 3090；原常驻加载实测 OOM，因此该入口默认
`E2_MEMORY_MODE=text-offload`：文本编码器和去噪模块交替暂存在主机内存，
计算仍为 CUDA bf16、原始 10 步。执行模式纳入模型身份，必须重新采集并通过 T0。
设 `E2_MEMORY_MODE=resident` 可选择原常驻方式，不会自动降低精度或步数。

```bash
bash scripts/reproduce_alpha_libero_e2_fresh.sh \
  --run experiment/e2_3090_fresh_NEW_RUN \
  --gpus 0 1 2 3 --deadline 2026-11-19T10:00:00+08:00
```

采集按任务分片，两张卡负责 Discovery、两张卡负责 Calibration；每个任务仍采集
10 个 episode，合并时验证完整覆盖、身份及文件哈希。后续 E2 和 Robustness
沿用按状态分片与严格合并，并在全部卡上验证 T0。模型加载串行，显存门槛
可用 `--minimum-free-mib` 指定（默认 18000 MiB，真实卸载预检峰值约 13 GiB）；
加载 OOM 最多尝试三次。

绝对截止覆盖预检、采集、加载等待、基线和扩展。配对阶段提前两分钟停止派发，
提前一分钟执行进程树清理与部分结果汇总，硬截止前兜底终止剩余进程。
查看 `parallel_status.json`、`parallel/logs/`、`extensions/status.json` 和
`final_status.json`。队列完成则提前退出，截止并不保证所有实验完成。

`scripts/reproduce_alpha_libero_e2_robustness.sh` 串行重新采集状态库、执行 T0、
冻结 Discovery 选择、计算特征及 Calibration、完成每个 split 的 2304 条 E2 基线，
然后调用原 robustness 启动器。原模型身份校验保持启用；原服务器结果仅供参照。
前置阶段没有时间上限，`HOURS` 仅限制最终 robustness 阶段，默认 10 小时。

迁移参照目录默认是 `experiment/reference/experiment/`，包含原
`h1h2_20261112_223445_299984/manifests/`、`h2_states_20261113_175408_491989/`
及 `e2_exploratory_20261113_205159_787207/`。可通过 `--reference-root` 指定其他位置。
本机默认直接读取 `/mnt/storage2/users/hycui_data/yyf/OpenWAM/assets/openwam_ckpt/`
`openwam_alpha/OpenWAM-Alpha-Sim-LIBERO`；启动前会对照原 manifest 核验所有模型资产哈希。
使用 `CKPT_DIR` 覆盖该目录。不会自动下载或复制模型。
若 `.cache/models/OpenWAM-Alpha-Sim-LIBERO/` 存在，优先使用该目录。
本次该目录的模型文件是指向上述本地资产的软链接，仅补入原目录缺失的
`download-manifest.json` 元数据，以保持模型资产清单一致。

```bash
mkdir -p experiment
setsid nohup env GPU=auto HOURS=10 bash scripts/reproduce_alpha_libero_e2_robustness.sh \
  > "experiment/e2_reproduction_launch_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
echo "后台启动 PID=$!"
```

入口默认使用 `.venv-openwam/bin/python`，可用 `RESEARCH_PYTHON` 覆盖。
本次本机环境在隔离 venv 中安装 research 依赖，并通过该环境内的
`shared_openwam_dependencies.pth` 只读引用本机已有 OpenWAM 环境的 PyTorch/CUDA 包；
该环境依赖共享目录持续可用，不可直接复制到其他服务器。
本机 robosuite 的 `macros_private.py` 关闭了写入共享 `/tmp/robosuite.log` 的文件日志，
保留启动器收集的控制台日志。

`GPU=auto` 选择无计算进程且空闲显存最多的物理 GPU，并固定其 UUID；
也可指定物理编号。GPU 不可用、资产不匹配、T0 失败或任一子命令失败都会停止后续阶段。
每次使用新的 `RUN` 目录，查看 `launcher.log`、`pipeline_status.json` 和
最终 `robustness/report.md`；达到时间预算不表示全部 7680 条配对已完成。

### 多卡接管与独立分片

`scripts/reproduce_e2_parallel.py` 可以接管上述流程的状态采集阶段。
保留已完成的状态库，由独立 worker 并行执行 T0、特征和后续配对；
E2 及 robustness 按状态序号取模分片，同一状态的全部方法和正负扰动在同一张卡上完成。
合并时检查计划、模型身份、分片归属、重复和缺项，原数值验证仍然启用。
robustness 的 10 小时是所有 worker 共享的截止时间，不是每张卡依次追加的时间。

本次工作池为 GPU 0、3、4、6；每次启动副本需至少 28000 MiB 空闲显存，
不足时等待，不停止其他任务。模型加载使用文件锁串行执行以限制主机内存峰值；
每个新 worker 默认使用 4 个 CPU 线程，可用 `E2_CPU_THREADS_PER_WORKER` 覆盖。
GPU 3、6 另执行 Discovery T0，配对 worker 在启动前检查这些额外证书。

运行状态见 `parallel_status.json`，主日志为 `experiment/e2_parallel_launch_*.log`，
各 worker 日志在运行目录的 `parallel/logs/`。原 `pipeline_status.json` 在接管完成后
标记为 `superseded`。分片保留在 `parallel/` 和 `robustness/workers/` 供追溯。

### 总计约十小时的开发任务扩展

`scripts/research_e2_extensions.py` 作为独立监督器等待原四卡流程完成，再运行预先固定的扩展队列：

- 按 Discovery 结果拟合每个层/步的分组敏感性排序；扰动低敏感、高敏感、均匀及五个随机分组对照。
  对照使用相同的 2/4/6 个分组预算，分别测试 gate=0 消息屏蔽与前一步 baseline V 的陈旧值替换。
- 用耦合动作/视频种子 44、45 重复 epsilon=.03 的分组敏感性实验及上述对照。
- 选取与原选择不重合的 saved episode ID，验证新状态上的敏感性和干预对照。
- 有余量时追加两种扰动方向和 .01/.03/.10 幅度扫描。

主比较为 Calibration 上低敏感分组相对五个随机对照平均值的动作 RMSE 差；
按 episode/task 分层 bootstrap，完整匹配全部八种策略后才纳入对照。
排序仅使用 Discovery；不同 episode 仍来自相同开发任务，不作为 held-out 泛化证据。
所有干预使用诊断计算路径，不代表实际缓存、token 删除、闭环成功率或加速收益。

启动监督器时显式设置整个实验的截止时间，例如：

```bash
setsid nohup .venv-openwam/bin/python scripts/research_e2_extensions.py \
  --run experiment/e2_reproduction_20261117_215825_3789091 \
  --core-pid 3897820 --deadline 2026-11-18T09:19:00+08:00 \
  > experiment/e2_extensions_launch.log 2>&1 < /dev/null &
```

与原启动器一样，需要已有的 LIBERO/PYTHONPATH/EGL 环境变量。
监督器写入 sealed `budget.json` 和 `extensions/protocol.json`；新 robustness worker
使用原截止时间和总预算截止时间中的较早者。扩展 worker 完成当前配对后在截止时间退出，
保留原始结果并生成 `extensions/report.json`、`report.md`。部分阶段明确标为不完整，
不通过额外重复运行来补满时间。查看 `extensions/status.json` 获取等待、运行和完成状态。
本次按用户选择从 11 月 17 日约 23:19 起再运行十小时，截止为 11 月 18 日 09:19（北京时间）。
尚未开始扩展结果计算时，可停止等待中的监督器并用 `--revise-deadline` 重启；
旧的 sealed 协议和预算保存在 `extensions/protocol_history/`，原实验不重启。

共享服务器的其他进程可能在启动检查后申请显存。worker 在取得模型加载锁后再次检查
28000 MiB 余量，模型加载阶段的 CUDA OOM 会清理分配并等待重试；推理和数值校验错误仍然停止。
若在 Calibration smoke 阶段失败，可使用 `reproduce_e2_parallel.py --resume-verified`
并指定 `--run`、`--gpus` 恢复：先验证完整 Discovery 的 2304 对、状态库和 T0，
归档失败记录和空 smoke 目录，然后仅重跑 Calibration 及后续阶段。
本次恢复工作池为 GPU 2、3、4、6，截止仍为 09:19，恢复信息见 `parallel/recovery.json`。
