# 后续投稿实验实现与执行记录

更新时间：2026-11-18 13:43，北京时间。

## 已实现和验证

实现位于 `benchmarks/action_cache/`，详细协议及命令见
`ACTION_CACHE_RESEARCH.md`。包含真实视频层级跳算、请求级缓存、固定/随机/
变化/静态敏感性/联合自适应刷新、直接缓存探针、原生算子分析、开发集筛选、
闭环任务清单冻结和配对分层统计。首帧与动作分支每层重算。

相关组合测试 **46 passed, 1 skipped**。跳过项是已有 CUDA 权重卸载测试，
原因是工具沙箱内没有 GPU。新缓存的 21 项测试均通过；真实模型缓存预检、
实际加速比和闭环成功率尚未产生，不能将 CPU 测试当作 GPU 验收。

原运行模型/研究源码保持不变，当前 E2 身份不受新模块影响。
新实现通过独立 source hash 固定；prepare 后修改新代码需要新的协议目录。

## 当前产物和任务

- 冻结的可行性协议：`experiment/action_cache_protocol_20261118/protocol.json`。
- E2 证据快照：同目录 `e2_evidence.json`。
- 后台预检等待队列原 PID：`2581035`；已按用户要求停止，不再自动启动后续实验。
- 队列目录：`experiment/action_cache_queue_20261118/`。
- 队列日志：`experiment/action_cache_queue_20261118.launch.log`。
- 队列截止：2026-11-19 09:58+08:00，早于原硬截止 10:00。

队列只在旧任务产生 `final_status.json`、候选 GPU 空闲且原 GPU 文件锁释放后
启动。截止前不足 30 分钟就不再启动模型，状态记为
`not_run_budget_reserved_for_existing_experiment`，不为跑新实验而延长预算。

预检固定比较 Native N10、refresh-all、fixed-2、fixed-3、change-0.1，
使用八个 Discovery 任务和策略种子 43；3090 明确使用 text-offload，
结果只作诊断，不作为大显存 resident 性能主表。

13:42 E2 快照：Discovery 2304/2304 已合并，Calibration 871/2304；
两 split T0 通过，Robustness 尚未开始，扩展监督器等待 core。
这个快照不是最终结果。

## 未完成的论文证据

1. 当前 E2 的 Robustness 和干预对照；直接复用误差与实际 GPU 跳算验收。
2. 独占 A100/H100 上 resident 算子分析及成功率—延迟对照。
3. Calibration 三种子筛选及闭环筛选后，冻结 24 个 heldout 任务，
   每任务 50 初始状态 × 3 策略种子；样本量需结合开发集功效分析锁定。
4. 最新相关工作与最强兼容外部方法的复现；目前未完成联网文献核查。
5. RoboTwin checkpoint/模拟器资产与第二架构模型的适配和验证；
   当前环境尚无已验证的第二平台，不伪装为已经支持。

新方法是否成立由上述结果决定；目前状态是“实现可运行、CPU 校验通过、
GPU 实验由用户按命令手动启动”，不是“已经得到顶会级结果”。
