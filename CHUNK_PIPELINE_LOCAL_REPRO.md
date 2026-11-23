# 本地四卡短测复现（2026-11-20）

## 本次结果

最新整轮屏障版本在物理 GPU **0,2,4,6** 完成短测，两组均 **20/20**。原生稳态客户端 P50/P95 为 **971.579/1305.390 ms**，屏障流水线为 **729.297/868.900 ms**；329 次稳态请求全部接管 stage=9、在线补 1 步。屏障等待 P50 为 **602.038 ms**，仍计入响应延迟。详情见 `experiment/chunk_pipeline_repro_20261120_170749/REPORT.md`。以下保留旧版本结果用于对照，不视为同一轮受控 A/B。

按要求使用物理 GPU **0,2,4,6** 重跑已完成（原生使用 0 卡），两组均为 **20/20**。稳态客户端延迟 P50/P95：原生 **950.286/957.999 ms**，四卡 **742.958/836.808 ms**；P50 降低约 **21.8%**（约 **1.28×**），P95 降低约 **12.6%**。运行中采样仅见本实验服务进程占用所选四卡，但不代表全程独占。详细结果见 `experiment/chunk_pipeline_repro_20261120_161345/REPORT.md` 和同目录 `comparison.json`。未启动覆盖或深度阶段。

此前 `1,2,3,0` 卡短测同样为 20/20，但延迟无明显改善且存在共享负载；保留记录于 `experiment/chunk_pipeline_repro_20261120_152907/REPORT.md`。

## 实现与内存策略

当前调度已改为整轮屏障：新反馈到达后，等待全部后台 worker 完成本轮预算（默认每个未来 chunk 推进 3 步，累计最多 9 步），才用真实观测更新头部并在线补步。正常接管不再提前取消后台任务；reset、条件签名变化和 close 仍可取消。默认在线补步应为 10、7、4、1、1…；此前实验属于提前接管版本，不能作为新版本结果。`background_wait_ms` 包含整轮屏障等待，`ready_stage_at_observation` 与新增 `ready_stage_at_takeover` 分别记录到达和接管进度，端到端计时不会扣除等待。此改动不保证总延迟降低。

根目录提供的 `chunk_pipeline.py`、`engine.py`、`server.py`、`scheduler.py` 已集成到部署与 LIBERO 模块，四个文件的指纹均与 `CHUNK_PIPELINE_REPRO_20261120.md` 一致。原始来源文件保留。

每个 worker 在自己的 CUDA 设备常驻完整模型和 chunk 状态。不启用文本编码器 CPU 卸载、CPU 推理、显存不足时的自动迁移或降级。启动器显式清除 `E2_MEMORY_MODE`；新启动的复现实验通过 `OPENWAM_REQUIRE_GPU_RESIDENT=1` 检查模型全部注册参数和缓冲区的设备。CPU 上的请求编排、模拟器、结果转换及原生随机数生成不属于模型卸载。

现有 Wan 预处理接口通过 `**kw` 接收 `pin_first_frame`、`initial_latents`，但不消费这两个额外参数。流水线自身保留已有 video/action latent，显式替换首帧并在每步重新固定首帧；当前骨干也会生成首帧条件与 TI2V 时间调制标志。兼容测试覆盖这些路径，因此没有改写视频骨干。

## 运行

```bash
GPU_LIST=0,2,4,6 bash scripts/repro_chunk_pipeline_smoke.sh
```

默认使用 `.venv-openwam` 作为服务环境、`.venv-libero` 作为模拟器环境；模型目录为 `/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO`，GPU 顺序为 `1,2,3,0`，端口为 `8993`。可通过 `CKPT_DIR`、`GPU_LIST`、`BASE_PORT`、`RUN` 覆盖。启动前应确保所选四卡显存足够且端口空闲；OOM 时停止，不启用 CPU fallback。

启动器核验权重哈希和环境，依次执行 native 与 chunk-pipeline：N10/H32/E8、seed 42、同步、无 compile/DiT cache、部署视频输出关闭。每组四个 suite 的 task 0 各跑 5 个初始状态，共 20 回合。流水线后台仍解码预测视频帧，每轮推测 3 步、累计不超过 9 步。

输出目录为 `experiment/chunk_pipeline_repro_<timestamp>/`，包含启动脚本、日志、源码指纹，以及两组各自的 manifest 和 summary。短测完成后可重新汇总：

```bash
.venv-openwam/bin/python scripts/summarize_chunk_pipeline_smoke.py experiment/chunk_pipeline_repro_<timestamp>
```

## 延迟与验收

- `OPENWAM_LIBERO_REQUEST_TIMING=1` 启用相同的客户端计时：从观测图像处理开始，到收到并转换动作为止。
- 每 8 个动作的第一个请求为 chunk 请求；包含冷启动和排除每回合前三个 chunk 的统计分别报告。两组都按相同规则筛选，不混用原生和流水线各自不同的服务端计时字段或 `cold` 定义。
- 汇总器验证两组完成相同 20 个 suite/task/trial，核对四卡 manifest，并检查计时数量与回合步数一致、四个 head worker 均出现。重试日志不自动混入延迟统计，需人工筛选后重新汇总。
- `comparison.json` 包含逐回合成功结果、客户端 P50/P95、稳态在线补步分布及 worker 显存峰值；服务端 action-ready 时间独立报告。
- 与文档的 20/20 参考结果比较，不把参考成功率或提速倍数视为本次必然结果。本次 GPU 常驻配置与文档的 text-offload 配置有明确差异。
- GPU 为共享资源；启动器保存进程快照。若同卡出现其他计算进程，延迟结果只能作为受干扰实测，不能据此确认原文提速倍数。

## 测试

```bash
.venv-openwam/bin/python -m pytest tests/test_chunk_pipeline.py tests/test_chunk_pipeline_repro.py tests/benchmarks/test_libero_scheduler.py tests/benchmarks/test_libero_bridge.py tests/benchmarks/test_libero_single_eval.py tests/test_deployment_changes.py -q
```

部署测试中已有 `TestCudagraphMarkStepBegin.test_mark_appears_before_not_after` 结构检查失败，涉及未修改的 `openwam/model/architectures/base.py`；本次未修复这一无关问题。
