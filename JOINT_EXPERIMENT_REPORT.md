# OpenWAM Joint 并行加速实验阶段汇报

数据快照：2026 年 11 月 6 日 16:00（Asia/Shanghai）。重复验证尚未完成，后续结果可能更新。

## 一、核心结论

1. **阶段并行具有明显加速潜力。** LIBERO 同轮固定输入回放中，Joint Native N10 的请求 P50 为 **965.25 ms**，Joint shared N10 为 **297.17 ms**，约 **3.25×** 加速。但该轮阶段批量 parity 检查失败，只能作为性能诊断，不是等价性证明。
2. **prepared 是当前优先推进方案。** RoboTwin 50 任务完整评测中，prepared 与现有 Joint 并行基线均达到 **90.0%** 成功率，各分片热请求 P50 进一步降低约 **13%–14%**。
3. **prefix 更快，但风险更高。** 同轮完整评测中，其 P50 降低约 **16%–21%**，成功率降至 **88.4%**，且一个分片出现明显 P95 长尾。
4. **尚不能宣称无损加速。** 总成功率持平不等于逐场景一致；重复验证已发现 prepared 的稳定退化个例。

## 二、实验目标与方案

目标是在当前 OpenWAM 仓库中参考 ParaFlow 的历史复用及执行优化方法，加速视频—动作联合推理，同时尽量保持任务成功率。

| 方案 | 定位 |
|---|---|
| Joint Native N10 | 原生 Joint 模式，10 个阶段串行执行 |
| Joint parallel / legacy | Joint 阶段并行实现，后续执行优化的参考后端 |
| Joint prepared | 在阶段并行基础上进一步复用预处理及条件准备 |
| Joint prefix | 进一步利用前缀复用，探索更高加速 |

### 统一口径

- RoboTwin 后续实验中的 **legacy 已经是阶段并行版本，不是 Native N10**。
- 阶段并行将多个阶段样本组成批量执行，不等于蒸馏后的 N1 模型。
- 当前 Joint 路径包含未来视频 **latent 去噪**，不包含可播放视频的 **VAE 解码**。
- 请求时延不是整个 episode 耗时，不包含完整仿真执行时间。
- 四张 GPU 用于评测任务分片，不代表单个请求使用四卡加速。
- 不跨 benchmark、checkpoint、实现版本或计时边界直接计算加速比。

## 三、LIBERO 全量质量评测

覆盖 Spatial、Goal、Object、Long 四个套件，共 **40 个任务，每任务每方案 50 回合**。每方案 2,000 回合，全部完成，三组均使用 E8。

| 方案 | 成功回合 | 成功率 | 相对 Native N10 |
|---|---:|---:|---:|
| Native N10/E8 | 1,974/2,000 | 98.70% | 基线 |
| Native N4/E8 | 1,980/2,000 | 99.00% | +0.30 个百分点 |
| Joint N4/E8 | 1,975/2,000 | 98.75% | +0.05 个百分点 |

### Joint N4 请求计时

| 阶段 | 请求数 | 请求占比 | 去噪 P50 | 完整请求 P50 | 完整请求 P95 |
|---|---:|---:|---:|---:|---:|
| 冷启动 | 4,000 | 9.8% | 370.28 ms | 399.90 ms | 560.16 ms |
| 稳定阶段 | 36,846 | 90.2% | 150.57 ms | **185.58 ms** | 362.68 ms |

结论：该轮 Joint N4 与 Native N10 的总体表现接近，90.2% 的请求进入稳定阶段。但 Native 没有记录该轮同口径逐请求时延，不能由这组闭环评测直接计算三方案加速比；冷启动时延也不是 Native 时延。

## 四、LIBERO Native 与阶段并行的同轮测速

条件：**NVIDIA A100 80GB、相同 LIBERO checkpoint 和真实观测 trace、H32/E8、预热 48 次、计时 120 次**。各方案均执行未来视频 latent 去噪，关闭 VAE 像素解码。

| 方案 | 去噪阶段数 | 完整请求 P50 | 完整请求 P95 | 相对 Native 的 P50 加速比 | P50 时延降低 |
|---|---:|---:|---:|---:|---:|
| Joint Native N10：串行 | 10 | **965.25 ms** | **1,307.15 ms** | 1.00× | — |
| Joint shared N10：阶段并行 | 10 | 297.17 ms | 311.97 ms | **3.25×** | 69.2% |
| Joint shared N4：阶段并行 | 4 | 156.04 ms | 170.85 ms | **6.19×** | 83.8% |

### 解释与限制

- N10 对比保留相同阶段数，但执行算法和轨迹不同，不意味着输出等价。
- N4 同时减少阶段数，不能将全部收益归因于并行。
- **阶段批量 parity 检查未通过**，阈值为 `rtol=atol=0.05`；以上是诊断性测速，不是数值等价或闭环质量保持的证明。
- 这轮固定输入测速与上一节的全量成功率评测不是同一次实验，不能拼成“同一版本已验证无损 6.19× 加速”的结论。
- 后续 RoboTwin prepared/prefix 数据来自不同 checkpoint 和工作负载，不能与本表连乘得到端到端加速比。

## 五、RoboTwin 全任务完整对比

覆盖 **50 个任务，每任务每后端 5 回合，共 750 回合**，全部完成。这是当前 prepared/prefix 优化最主要的质量证据。

### 成功率

| 后端 | 成功回合 | 成功率 | 相对 legacy | 配对退化 / 改善 |
|---|---:|---:|---:|---:|
| Joint Native N10 | 未进行该轮 250 回合评测 | — | — | — |
| Joint parallel N10 / legacy | 225/250 | **90.0%** | 基线 | — |
| Joint prepared N10 | 225/250 | **90.0%** | 0 个百分点 | 9 / 9 |
| Joint prefix N10 | 221/250 | 88.4% | −1.6 个百分点 | 11 / 7 |

配对退化表示相同场景下 legacy 成功、候选失败；改善则相反。prepared 的总成功数持平，但存在 9 个退化和 9 个改善，不是逐场景一致。

### 各分片闭环热请求时延

| 分片 | legacy P50 | prepared P50 | prepared 时延降低 | prefix P50 | prefix 时延降低 |
|---|---:|---:|---:|---:|---:|
| 0 | 354.89 ms | 308.99 ms | 12.9% | 286.17 ms | 19.4% |
| 1 | 360.83 ms | 309.14 ms | 14.3% | 285.92 ms | 20.8% |
| 2 | 356.77 ms | 308.27 ms | 13.6% | 300.85 ms | 15.7% |
| 3 | 359.70 ms | 311.95 ms | 13.3% | 291.16 ms | 19.1% |

- 这是各分片闭环请求统计，不是同输入配对测速；不合并或平均分片 P50。
- prefix 分片 2 的 P95 为 **742.43 ms**，同分片 legacy 为 **373.04 ms**，长尾原因尚未确认。
- RoboTwin Native 的小规模评测没有记录可用的同口径 Native 请求计时，因此本表不填入推测值。

### 同输入固定观测回放：补充诊断证据

另一轮 RoboTwin 回放预热 5 次、计时 10 次，得到：

| 后端 | P50 | P95 | 相对 legacy 的 P50 加速比 |
|---|---:|---:|---:|
| legacy | 345.13 ms | 365.00 ms | 1.00× |
| prepared | 314.49 ms | 330.54 ms | 1.10× |
| prefix | 289.89 ms | 302.57 ms | 1.19× |

该轮整体标记为 `diagnostic_only_parity_failed`。prepared 的部分输出／算子检查通过不代表全部验证通过；prefix 存在内部预测超阈值差异。此表不测闭环成功率，也不替代前述完整任务评测。

## 六、Joint 与 isolated 的匹配权重诊断

覆盖 `adjust_bottle`、`open_laptop`、`stack_blocks_two`，每任务每方案 5 回合。

| 方案 | 成功回合 | 成功率 |
|---|---:|---:|
| Joint Native N10 | 14/15 | 93.33% |
| Joint parallel N10 | 14/15 | 93.33% |
| isolated Native N10 | 15/15 | 100.00% |
| isolated ParaFlow N10 | 15/15 | 100.00% |

Joint 与 isolated 使用各自匹配的训练权重。样本较少且权重不同，不能据此证明 isolated 整体优于 Joint，或将差异归因于执行优化。
早期将 mutual 权重强制切换 isolated mask 后全部失败的结果属于权重／mask 不匹配消融，不应作为原版 isolated 的效果。

## 七、为什么不能直接以 ParaFlow 的 110 ms 为对照

历史路径复现结果：

| 路径 | P50 | P95 |
|---|---:|---:|
| first-frame controlled | **111.45 ms** | 122.14 ms |
| full-video / action history | 343.64 ms | 347.24 ms |
| full-video / phase history | 343.63 ms | 347.40 ms |

- 约 110 ms 的路径没有执行 full-video forward，不能与包含未来视频 latent 去噪的 Joint 直接比较。
- 增加 full-video 的两条路径仍使用 isolated checkpoint，且执行路径不同，不是 mutual Joint 的同工作负载对照。
- 该扩展实验只做了重复性与有限值检查，没有闭环质量验证。
- 路径差值不能解释为纯视频算子耗时，也不能据此宣称 Joint 比 ParaFlow 慢多少倍。

## 八、差异场景重复验证：当前快照

从完整实验中选取 **29 个差异场景、17 个任务，每场景每后端重复两次**，预期 174 回合。因为是差异筛选集，其成功率不能外推为整体 benchmark 成功率。

截至本报告快照，完成 **165/174 回合**，尚缺 9 回合：

| 后端 | 有效完成 / 预期 | 已完成中的成功回合 |
|---|---:|---:|
| legacy | 58/58 | 33 |
| prepared | 56/58 | 37 |
| prefix | 51/58 | 33 |

完成数量不同，不直接比较各自原始成功率。只比较三者均完成的 **49 个共同样本**：

| 后端 | 共同样本成功回合 | 共同样本成功率 |
|---|---:|---:|
| legacy | 28/49 | 57.1% |
| prepared | **34/49** | **69.4%** |
| prefix | 31/49 | 63.3% |

分别与 legacy 配对：

| 候选 | 配对数 | legacy 成功 | 候选成功 | 退化 | 改善 |
|---|---:|---:|---:|---:|---:|
| prepared | 56 | 32 | 37 | 5 | 10 |
| prefix | 51 | 29 | 33 | 6 | 10 |

两行配对集合不同，不可直接用来排序候选方案。未完成回合不计为失败。

### 风险发现

- legacy 的 29 个场景中，3 个在两次重复间出现成功／失败变化，基线存在一定波动。
- `blocks_ranking_rgb_e2_7295d070571f` 中，legacy 两次均成功，prepared 两次均失败，存在稳定候选退化。
- 基线随机性不能自动解释所有候选退化；还需通过相同观测及历史条件进行归因。
- 当前续跑使用共享 GPU 模式，只补闭环正确率，跳过独占测速；共享负载下的闭环时延不用于正式性能对比。

## 九、下一步与汇报建议

1. 补齐当前重复实验剩余回合，生成完整配对质量结论。
2. 优先分析 prepared 的稳定退化场景，在同输入、同历史、同随机状态下定位最早预测／动作偏差。
3. 独占 GPU 上增加 RoboTwin **Native、legacy、prepared、prefix** 同轮固定输入测速，统一权重、观测、预热及计时边界。
4. 分别报告输出／算子／批量分区验证和 P50/P95，不把“可运行”“更快”或“总成功率持平”等同于无损。
5. prefix 的质量差异和长尾未解释前，不升级为默认方案，也不继续叠加更复杂优化。

### 可直接引用的汇报结论

> 我们完成了 Joint 阶段并行及后续执行优化的阶段性验证。LIBERO 全量评测中，Joint N4 成功率为 98.75%，与 Native N10 的 98.70% 接近。另一次同输入回放中，Native N10 请求 P50 为 965 ms，N10 阶段并行为 297 ms，获得约 3.25× 的诊断性加速，但尚未通过全部数值一致性验证。在 RoboTwin 50 任务完整评测中，prepared 在保持 90.0% 总体成功率的同时，将各分片热请求 P50 进一步降低约 13%–14%；prefix 加速更大，但成功率降至 88.4%，且存在长尾风险。目前将 prepared 定位为优先候选，继续完成稳定退化归因和独占同输入测速，尚不宣称无损或跨 benchmark 的端到端总加速比。

## 十、数据来源

路径相对于仓库根目录：

- LIBERO 全量质量与阶段计时：`outputs/libero/compare_joint_full/comparison_timing_accuracy.md`
- LIBERO Native／阶段并行同轮测速：`outputs/joint_native_parallel_20261031_171959/timing.json`、`comparison.md`
- RoboTwin 完整 50 任务：`outputs/robotwin/joint_three_backend_50tasks_e5_4gpu_v1/shard_{0,1,2,3}/comparison.json`
- RoboTwin 小样本同输入回放：`outputs/robotwin/joint_execution_replay_20261102_162750/timing.json`
- Joint／isolated 匹配权重诊断：`outputs/robotwin/joint_isolated_diagnostic/comparison.json`
- ParaFlow 历史路径：`outputs/paraflow_110_20261031_135019/measurement/video_summary.json`
- 重复验证：`outputs/robotwin/joint_quick_5h_v1/suite.json` 及各分片有效结果；本节数字按本报告时间点重新校验。
- 动态汇报入口：`bash scripts/finish_joint_report.sh summary`，更新 `outputs/robotwin/joint_quick_5h_v1/reporting/comparison.md`。本文件是静态阶段快照，不会自动刷新。
