# 原生 N10 / 当前优化 chunk：完整闭环评测

正式矩阵：LIBERO 四套件40任务×50回合×2方案，以及 RoboTwin demo_clean
50任务×100回合×2方案，共14,000回合。预检另计70回合。
只使用物理GPU 0、2、4、6；禁止CPU卸载和显存不足后的CPU回退。

## 启动、恢复和汇总

```bash
.venv-openwam/bin/python scripts/run_native_chunk_comparison.py --benchmarks robotwin libero --detach
```

默认输出 `experiment/native_chunk_full_20261121`。同一命令恢复，已完整通过的任务/方案不重跑。
`--detach` 创建独立进程会话，终端退出后继续，管理器日志写入 `manager.log`。
`--run` 指定新输出目录；`--preflight-only` 只验证接口和计时；`--summary` 重建报告。
恢复必须使用相同任务顺序、源码、权重、配置和端口。输出目录持有排他锁。
不要在实验运行期间修改实验所用源码：指纹变化会停止运行，避免混用版本。

## 对比口径

- `native_n10`：真正的串行原生 N10，不是前一轮线程版 chunk 基线。
- `chunk_optimized`：常驻进程、CUDA Graph、完整VAE图和已验证的选帧/缓存优化。
- 两方案均 H32/E8、N10、相同精度和各自基准权重，关闭 compile/DiT cache/输出视频。
- 优化版每轮3步后台去噪，累计9步加1步在线，保留全worker等待屏障。
- Native 单请求一卡；优化单请求四卡；计时不混跑，不能将请求加速称为四卡吞吐提升。
- 固定封存manifest后配对初态、环境随机域和指令；逐任务交替方案顺序。
- 每回合请求分为首次、前3次中的后2次、稳态推理和缓存动作；缓存不稀释推理时延。
- 回合计时包含reset，rollout不含reset；LIBERO和RoboTwin环境生命周期不同，不跨基准比较。
- RoboTwin基准视频录制在两方案中均关闭，保留正常观测渲染和原有仿真动作执行。

## 产物与运行保护

- `STATUS.md` / `state.json`：当前单元、最近回合进度和基于正式样本的剩余时间估算。
- `REPORT.md` / `summary.json`：正式成功率、配对退化/改善、分阶段请求/回合耗时及加速比。
- `tasks.csv` / `episodes.csv`：任务和逐回合原始指标；最终报告提供配对bootstrap区间。
- 每单元 `requests.jsonl` / `stages.jsonl`：客户端请求与服务端阶段指标，一一核对推理次数。
- `gpu-monitor.jsonl` / `resources.json`：GPU利用率、显存峰值和采样估算的GPU占用秒数，不等同精确计费时间。
- `protocol.json`：源码、仿真源码、权重及配置指纹；manifest附带初态和任务资源指纹。
- 各次attempt保留命令、失败及原始日志。失败最多重试两次；30分钟日志无进展或单任务超过24小时则清理自有进程。
- 外部GPU活动使本次attempt无效，等待后重试；磁盘剩余不足5GiB停止，不删除其他文件。
- manager退出通过所有权lifeline清理子进程，不按进程名称批量kill。

成功率差异不阻止正式对比，但不能据此宣称无损或自动切换默认部署配置。
只有正式14,000回合完整、场景匹配且日志有效才能报告全量完成。
