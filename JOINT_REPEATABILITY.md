# Joint 加速阶段汇报与重复验证

## 三小时收尾与自动对比汇报

执行 `bash scripts/finish_joint_report.sh`：检查 GPU 4–7 无其他计算进程、磁盘至少剩余 5 GiB 后，自动创建 `joint_report_3h` tmux 会话。
沿用 `outputs/robotwin/joint_quick_5h_v1` 的两次重复、零对照配置，跳过有效结果；完成闭环后执行固定输入回放（预热 8、计时 32、profiling 2）。
评测及回放合计限时 `180m`，停止时最多额外等待 60 秒。不会终止其他实验；无法访问 GPU、GPU 忙或空间不足时拒绝启动。

结束或超时后生成 `outputs/robotwin/joint_quick_5h_v1/reporting/comparison.md` 与 `comparison.json`：完整来源实验、当前有效完成数量、三者共同样本成功率、配对退化／改善、稳定退化场景，以及各分片回放 P50/P95 和验证标签。
未完成回合不计为失败，差异筛选集不作为总体成功率，不使用历史回放冒充本轮数据；parity 失败只作诊断测速。
只更新汇报、不运行 GPU：`bash scripts/finish_joint_report.sh summary`。前台执行：`bash scripts/finish_joint_report.sh run`。
可通过 `REPEAT_ROOT` 指向同配置的既有实验，通过 `SESSION` 修改会话名称；已有回放参数不得修改以免破坏续跑身份。

### 与其他进程共享 GPU

执行 `ALLOW_SHARED_GPU=1 bash scripts/finish_joint_report.sh`，允许 GPU 4–7 上存在其他计算进程。
共享模式只续跑闭环正确率，跳过要求独占的固定输入测速，仍限时三小时并生成汇报；不会停止或修改其他任务。
启动前要求每张卡至少空闲 32 GiB，可通过正整数 `MIN_FREE_GPU_GIB` 调整。这是瞬时空闲检查，不是显存预留或不发生 OOM 的保证；共享竞争也可能使三小时内完成的回合变少。
`reporting/shared_gpu.json` 记录共享模式、启动前进程和显存快照；汇报明确注明共享闭环时延不可用于正式性能对比。
之后 GPU 空闲时，不设置 `ALLOW_SHARED_GPU` 再运行同一脚本，即可跳过已完成闭环并执行独占回放。

数据截止：2026-11-03。本文件描述已落盘结果及后续实验，不表示新实验已经启动。

## 阶段结果

### LIBERO

全量评测覆盖 40 个任务、每任务 50 回合，每方案 2000 回合：

| 方案 | 成功回合 | 成功率 |
|---|---:|---:|
| Native N10/E8 | 1974/2000 | 98.70% |
| Native N4/E8 | 1980/2000 | 99.00% |
| Joint N4/E8 | 1975/2000 | 98.75% |

Joint N4 稳定阶段请求 P50 为 185.58 ms，冷启动为 399.90 ms；Native 没有同口径逐请求计时，不能据此计算三者的可信加速比。

早期 N10 单任务诊断中，Native 为 5/5，shared Joint 为 2/5；phase shared/unshared 均为 2/5，action-history 变体为 0/5，tail-bridge 变体为 1/5 至 3/5。它们是早期实现的小样本诊断，不能替代全量结论或与后续版本合并计算。

在 mutual 权重上强制切换 isolated mask 的 0/5 结果属于 mask 不匹配消融，不是匹配权重下的原版 isolated 结论。

来源：`outputs/libero/compare_joint_full/comparison_timing_accuracy.md`、`outputs/libero/compare_n10_20261031_205720/comparison.md`、`outputs/libero/joint_quality_20261031_215338/comparison.md`、`outputs/libero/joint_tail_20261031_231528/comparison.md`、`outputs/libero/isolated_ablation_20261101_004138/comparison.md`。

### ParaFlow 历史基准

| 路径 | P50 | P95 |
|---|---:|---:|
| first-frame controlled | 111.45 ms | 122.14 ms |
| full-video / action history | 343.64 ms | 347.24 ms |
| full-video / phase history | 343.63 ms | 347.40 ms |

111 ms 路径没有 full-video forward。后两行增加了未来视频 latent 计算，但仍使用 isolated checkpoint，且执行路径不同，不是当前 mutual Joint 的同工作负载对照。只完成重复性/有限值检查，没有闭环质量验证；差值不能解释为纯视频算子耗时。

来源：`outputs/paraflow_110_20261031_135019/measurement/video_summary.json`。

### RoboTwin 匹配权重诊断

三任务、每任务 5 回合：Joint Native 与 Joint parallel 均为 14/15，isolated Native 与 isolated ParaFlow 均为 15/15。不同训练权重不能用于跨 mask 的因果加速或成功率归因。

来源：`outputs/robotwin/joint_isolated_diagnostic/comparison.md`。

### 最新 RoboTwin 全任务闭环

50 个任务、每任务每后端 5 回合，全部 750 回合完成。这里的 legacy 是 Joint 阶段并行的原执行后端，不是 Native N10。

| 后端 | 成功回合 | 成功率 | 相对 legacy 配对退化 / 改善 |
|---|---:|---:|---:|
| legacy | 225/250 | 90.0% | — |
| prepared | 225/250 | 90.0% | 9 / 9 |
| prefix | 221/250 | 88.4% | 11 / 7 |

| 分片 | legacy 热请求 P50 | prepared 热请求 P50 | prefix 热请求 P50 |
|---|---:|---:|---:|
| 0 | 354.89 ms | 308.99 ms | 286.17 ms |
| 1 | 360.83 ms | 309.14 ms | 285.92 ms |
| 2 | 356.77 ms | 308.27 ms | 300.85 ms |
| 3 | 359.70 ms | 311.95 ms | 291.16 ms |

prepared 各分片 P50 降低约 13%–14%；prefix 约 16%–21%。这是分片闭环统计，不是合并 P50 或固定输入配对测速。prefix 分片 2 的 P95 为 742.43 ms，legacy 为 373.04 ms，长尾尚未归因。

prepared 总成功率持平不等于逐场景一致或已证明无损。较早的另一轮 60 回合实验为 legacy 59/60、prepared 57/60，也应保留为质量风险证据，不能与当前实验混合统计。

来源：`outputs/robotwin/joint_three_backend_50tasks_e5_4gpu_v1/shard_0/comparison.json` 至 `shard_3/comparison.json`，以及 `outputs/robotwin/prepared_quality_60pairs_20261101_203307/comparison.md`。

### 固定观测回放及语义边界

一轮 5 次预热、10 次计时回放的 P50：legacy 345.13 ms，prepared 314.49 ms，prefix 289.89 ms。该轮 prepared 的 15 个请求最终输出与同输入算子检查通过，但参考批量/逐项检查失败；prefix 内部预测存在超阈值差异，故整体仍标记为诊断数据。

来源：`outputs/robotwin/joint_execution_replay_20261102_162750/`。

当前结论：保留 legacy 为参考，prepared 为优先候选，prefix 为实验候选。不能宣称比 isolated 更快，也不能将批量/逐项差异等同于优化算子错误。Joint 包含未来视频 latent 去噪，未包含可播放视频的 VAE 解码；一次阶段批量前向不是蒸馏后的 N1。四卡用于评测分片，不降低单请求时延。

## 新实验：先归因，再优化

`benchmarks.robotwin.joint_repeatability` 从原始 sealed manifest 和逐任务结果提取差异，不依赖汇总表的成功率。当前来源得到 29 个差异场景（17 个任务），再选择 10 个三个后端均成功的对照场景；对照按任务名、episode 排序，每任务最多一个。

- 39 个场景 × 3 个后端 × 5 次重复 = **585 回合**。
- 原始 seed、完整 seen/unseen 指令、source episode、manifest hash 都被保留；单场景子 manifest 的 episode 重编号为 0，原始编号在 provenance 中保留。
- 场景确定性分片，同一场景所有重复和后端在同一 GPU。每个重复以固定随机种子打乱后端顺序；为避免数百次权重加载，一个后端块复用一个服务端，每回合由现有客户端 reset 清理状态。
- legacy 的重复结果用于描述基线波动；候选与同轮 legacy 配对，分别报告退化、改善和未解释差异。所选场景有选择偏差，不报告为全测试集成功率。
- repeat 0 记录真实请求输入和输出动作，便于寻找首个观测/动作分歧；其余重复不保存原始请求，以控制磁盘占用。
- 重复阶段全部完成后，自动在四卡上分别进行固定输入回放：48 预热、120 计时，另做 8 个样本的 profiler。三个后端共享输入，计时阶段克隆参考历史和 RNG，独立历史验证在计时之外。新入口启用 `--validate-every-request`，每个验证请求重新检查冻结候选，不仅复用之前同形状检查的结果；这部分可能耗时较长，但不计入测速。
- 不修改模型、mask、N10/H32/E8 或 0.05 的现有误差阈值。允许已知 parity failure 后继续，但始终标记诊断性质，不自动改变默认后端。
- profiler 以正式回放中 prefix 最慢的 8 个请求为目标，在独立的 instrumented pass 中对三个后端使用相同请求索引。它是长尾归因线索，不保证复现历史分片 2 的同一负载异常。

### 一条指令，自动 tmux 和四卡

先确认 GPU 0、1、2、3 可用：

```bash
GPUS=0,1,2,3 bash /home/zhijian/OpenWAM-main/scripts/validate_joint_repeatability.sh
```

默认创建 `joint_repeatability` tmux 会话，交互终端会自动进入。`Ctrl+b` 然后 `d` 仅退出界面，不停止实验；重新进入：

```bash
tmux attach -t joint_repeatability
```

同名会话存在时不会重复启动；完成后会话保留终端。若要续跑已结束/中断的实验，可在该会话内执行 `bash scripts/validate_joint_repeatability.sh all`。固定输出目录支持逐场景、逐重复、逐后端续跑；只有结果与 suite identity 都有效的回合才跳过。

默认端口为 8950–8953。可用 `GPUS`、`BASE_PORT`、`REPEATS`、`CONTROL_CASES`、`REPEAT_ROOT`、`SOURCE_ROOT` 调整，或通过 CLI 参数覆盖。修改场景选择、GPU 分配、代码、权重或实验参数后须使用新输出目录。不要在同一目录运行两份任务。

### 独立阶段与查看结果

```bash
bash scripts/validate_joint_repeatability.sh plan
bash scripts/validate_joint_repeatability.sh run
bash scripts/validate_joint_repeatability.sh replay
bash scripts/validate_joint_repeatability.sh summary
```

`plan` 只检查来源、选择场景并显示预算，不创建实验目录，也不加载模型或验证 GPU。`run` 只做闭环重复和动作分歧分析；`replay` 仅做固定输入验证、测速和 profiler。`all` 顺序执行两阶段。`analyze` 可单独重算已捕获请求的动作分歧。

默认输出为 `outputs/robotwin/joint_repeatability_v1/`：

- `suite.json`：不可变场景选择、来源哈希、重复次数、GPU/端口映射。
- `report.md` / `report.json`：全分片进度、配对结果、legacy 自身波动、逐场景分类。
- `cases/*/source.json`：原始 seed、指令与 manifest provenance。
- `shard_*/runs/*/repeat_*/{legacy,prepared,prefix}/`：结果、完成标记、attempt 日志、progress 和视频。
- `capture_diagnostics.json`：首个观测分歧、动作分歧和执行动作超阈值位置。闭环历史不同，不能用它替代同输入算子验证。
- `shard_*/replay/validation.jsonl`：最终输出、同输入算子、批量/逐项三个独立检查。
- `shard_*/replay/timing.md`、`profile.md` 和 profiler JSON：独占回放性能与单独的算子归因。
- `shard_*/hardware_{before,after}.json`：闭环负载快照。闭环日志计时含验证和部分 trace 开销，不能当正式性能结论。

用 `tail -F outputs/robotwin/joint_repeatability_v1/run_shard_*.log` 查看进度。

### 失败与解释规则

- 无效/缺失结果不当作任务失败，不计入成功率，重新执行该回合；原 attempt 保留。
- 同输入检查、批量/逐项检查、最终轨迹检查分别列出，整体严格 gate 不被放宽。
- legacy 重复有成功/失败变化，标记 `baseline_variable`，不是自动认定优化无责。
- legacy 全成功、候选全失败，标记 `repeatable_candidate_regression`；其他配对损失标记未解释差异。
- 同卡外来进程导致独占回放 guard 拒绝时，不终止外来进程；检查负载后重新执行 `replay`。失败回放会归档，成功回放不重跑。
- 每个服务端 trace 默认最多 10000 请求，达到上限后仅停止记录，不中断策略；若记录被截断或数据不足，回放拒绝生成完整性能结论。若需要更高上限，使用新目录并传 `--trace-limit`。
- 模拟器没有结构化暴露的终止原因标记为未知，不从 success/failure 猜测根因。保存最后 step 和原始日志供检查。
- 捕获 RGB、状态及动作会额外占用磁盘和时间；不保证一夜完成。已有结果不会被删除，不主动终止任何非本实验进程。

## 下一步决策

若 prepared 同输入回放出现可重复差异，先修复，不继续叠加优化。若主要差异来自闭环波动，再设计独立的广覆盖确认，不能用差异筛选集证明整体非劣性。prefix 在预测误差、退化场景和长尾未解释前不升级默认；后续算子融合或图优化须逐项消融，并分别通过正确性与性能门槛。
