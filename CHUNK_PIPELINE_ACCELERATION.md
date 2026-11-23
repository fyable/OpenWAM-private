# 四卡提交优化实验

入口（在项目根目录执行）：

```bash
.venv-openwam/bin/python scripts/run_chunk_acceleration.py
```

固定物理卡0/2/4/6；需要至少12个可用CPU，服务和回放限制在相同8个CPU，
模拟器另用4个CPU并固定CPU算子线程数1。目标卡有外部计算进程时等待；
各实验的PyTorch interop线程数固定1；服务intraop为4或1。
运行中采样到干扰会作废整组并重跑，最多3次。约1秒监控不保证绝对独占。
完整实验包括多次模型加载、288组微基准和80或100回合评测，可能需要数小时。
额外的真实请求采集及进程冒烟回合不进入性能统计。

## 受控回放

先通过真实LIBERO spatial/task0/trial0采集经过服务端预处理的请求，保存为`request.pt`。
固定这一请求，在GPU生成阶段0/3/6/9的latent、动作、条件快照及参考输出。
这些CPU快照仅用于回放输入，不是模型卸载；所有去噪仍在常驻GPU模型上执行。
阶段0/3/6各测三步，阶段9测一步；恢复和同步结束后计时，逐步同步与原实现一致。
控制输入一致，但不代表覆盖了所有真实/预测条件或所有阶段。

比较thread4、thread1、process1，每配置包含四张卡的单worker及四种三卡组合，
每组热身10次、测量30次、重复3轮，配置顺序轮换。记录单步/最慢完成时间、批次吞吐、
CPU进程时间、线程数和GPU频率。线程模式的进程CPU时间不能跨worker求和。
视频和动作均检查逐位一致性、最大绝对误差、RMSE及`rtol=1e-3, atol=1e-4`；不自动放宽容差。

多进程接入门槛：所有三步阶段和所有三卡组合，最慢完成时间P50相对thread1降低至少15%，
P95不恶化超过5%，且所有数值检查通过。原始JSONL保留每次测量，不将kernel活跃时间当成可实现时延。

## 可选部署后端

`inference.chunk_worker_backend=thread|process`，默认`thread`，可通过现有部署覆盖参数选择。
process模式主进程持有首卡模型，三个spawn子进程各自持有其余卡模型；chunk状态不跨进程搬运。
IPC粒度为在线完成、预测解码或后台整轮；在线动作先返回，之后再解码。完整屏障、
N10/H32/E8、每轮三步及累计九步均保持不变。无CPU offload、OOM fallback、环境睡眠、compile或DiT cache。
重置采用回合标识、停止并排空任务；工作进程出错直接报错，不静默回退。
`OPENWAM_CHUNK_DIAGNOSTICS=1`支持分进程日志；详细all-thread trace仅支持thread后端。
`OPENWAM_CHUNK_CAPTURE_PATH`仅用于首个真实请求采集；含图像和本地PyTorch快照，只加载可信本地文件。

## 端到端验收

A=thread4，B=thread1，C=process1。先跑A1/B1；微基准过门槛才运行C冒烟及C1。
每组四个suite的task0各5个初始状态，seed42；结果保存初始状态字节哈希并逐回合核对。
合格候选须20/20成功、P95相对A1不恶化超过5%；其中P50最低者胜出，平局优先B。
随后按候选2、A2反序复测；两次比较都要求P50降低至少10%、P95不恶化超过5%、20/20成功。
没有合格候选则停止复测并报告未通过，不修改默认配置。

输出在`experiment/chunk_acceleration_<时间>/`：`REPORT.md`、`acceptance.json`、
`micro-gate.json`、源码/权重指纹、初始状态哈希、逐配置日志和GPU监控。
置信区间采用2000次回合级配对bootstrap；不是完整LIBERO准确率评测，也不宣称逐位策略等价。
端到端统计还包含在线完整完成调用、在线step主机调用、预测解码和后台完整任务时间；
在线step主机调用不是纯GPU时间，不能与这些嵌套区间相加。初始状态哈希仅在
`OPENWAM_LIBERO_STATE_HASH=1`时增加，普通评测结果结构不变。
