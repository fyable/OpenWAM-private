"""Generate the final Chinese report from retained raw campaign artifacts."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiment'


def main():
    output = EXPERIMENT / 'chunk_optimization_final_20261121'
    output.mkdir(exist_ok=True)
    campaigns = ('chunk_acceleration_20261120_203804', 'chunk_autonomous_20261121',
                 'chunk_graph_repair_20261121', 'chunk_coverage_20261121', 'chunk_regression_20261121',
                 'chunk_shadow_20261121', 'chunk_integrated_20261121', 'chunk_final_confirmation_20261121')
    payload = {'campaigns': {}, 'coverage': [], 'default_changed': False}
    for name in campaigns:
        directory = EXPERIMENT / name
        state = directory / 'state.json'
        acceptance = directory / 'acceptance.json'
        payload['campaigns'][name] = {
            'state': json.loads(state.read_text()) if state.exists() else None,
            'acceptance': json.loads(acceptance.read_text()) if acceptance.exists() else None}
    for path in sorted((EXPERIMENT / 'chunk_coverage_20261121').glob('tasks-*.json')):
        payload['coverage'].append({'name': path.stem, **json.loads(path.read_text())})
    parity_path = EXPERIMENT / 'chunk_regression_20261121/full-trajectory-prompts-attempt1/evaluation/results.json'
    if parity_path.exists():
        rows = json.loads(parity_path.read_text())
        payload['trajectory_parity'] = {'prompts': len(rows), 'checks': sum(len(step) for row in rows for step in row['checks']),
                                       'exact': all(check['exact'] for row in rows for step in row['checks'] for check in step.values())}
    server_logs = EXPERIMENT / 'chunk_shadow_20261121/real-closed-loop-shadow-attempt1/evaluation/logs/servers'
    shadow = [line for path in server_logs.glob('*') for line in path.read_text().splitlines() if '[SHADOW_PARITY]' in line]
    payload['closed_loop_shadow'] = {'forwards': len(shadow), 'nonexact': sum('False' in row for row in shadow),
                                     'latency_valid': False}
    coverage = payload['coverage']
    payload['coverage_totals'] = {'baseline_successes': sum(row['baseline_successes'] for row in coverage),
                                  'candidate_successes': sum(row['candidate_successes'] for row in coverage),
                                  'trials': 40 * len(coverage)}
    payload['broad_success_gate_passed'] = (
        payload['coverage_totals']['candidate_successes'] >= payload['coverage_totals']['baseline_successes'])
    try:
        payload['final_gpu_processes_csv'] = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True)
    except Exception as error:
        payload['gpu_check_error'] = repr(error)
    (output / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# 四卡持续优化实验总结', '',
             '预算：2026-11-21 00:33:38至20:33:38（Asia/Shanghai）；物理GPU 0/2/4/6。',
             '固定N10/H32/E8、后台三步、完整等待屏障、原始权重/精度，无CPU模型卸载、无环境睡眠。', '',
             '## 结论', '',
             '- 常驻进程消除多线程提交竞争；固定形状forward图、完整VAE图、GPU拼接和选帧转换进一步降低时延。',
             '- task0四套件各五初态确认通过；160回合广覆盖成功数154对155，严格成功率门槛未通过，因此保持默认关闭。',
             '- 数值验证不代表全部闭环成功率等价，不将筛选通过替代最终确认，也不丢弃失败回合。', '',
             '## 各阶段原始结果', '', '| 实验/阶段 | 有效 | 成功 | 稳态P50 ms | P95 ms |',
             '|---|---|---|---|---|']
    for name, campaign in payload['campaigns'].items():
        state = campaign['state']
        if not state:
            continue
        for stage, row in state['stages'].items():
            metrics = row.get('metrics', {}).get('client_chunk_requests_excluding_first_three')
            if metrics:
                lines.append(f'| {name}/{stage} | {row.get("valid")} | {row.get("successes")} | '
                             f'{metrics["p50_ms"]:.3f} | {metrics["p95_ms"]:.3f} |')
    lines.extend(['', '## 广覆盖', '', '| task组 | 线程成功/40 | 图优化成功/40 | P50 ms | P95 ms |', '|---|---|---|---|---|'])
    for row in coverage:
        metric = row['candidate_metrics']['client_chunk_requests_excluding_first_three']
        lines.append(f'| {row["name"]} | {row["baseline_successes"]} | {row["candidate_successes"]} | '
                     f'{metric["p50_ms"]:.3f} | {metric["p95_ms"]:.3f} |')
    lines.extend(['', '## 数值与限制', '',
                  '- 图修复单开关及组合301项真实像素/条件/一步结果精确相同；四卡固定状态微基准通过。',
                  f'- 完整十步轨迹：{payload.get("trajectory_parity")}。',
                  f'- 真实闭环影子forward：{payload["closed_loop_shadow"]}；此轮不计时延结果。',
                  '- task1/2原线程38/40、图优化37/40；普通进程eager复测38/40、图复测37/40。',
                  '- libero_goal/task1/trial1在普通进程及图模式都失败，单次线程成功；尚未定位全部闭环差异，不以随机性作结论。',
                  '- 帧优化始终完整解码九帧，不切割因果latent。CUDA图输入形状签名检查，显存余量不足直接拒绝，无CPU回退。',
                  '- 早期外部GPU监控在自有进程退出瞬间出现误判，相关阶段保留为无效；后续监控用已记录PID启动身份识别退出残留。',
                  '- 计划24回合扩展被LIBERO固定五初态协议拒绝；失败保留，后改40回合/组，不修改协议。',
                  '- 单元/回归128通过、1个既有无关测试排除；GPU数值与闭环验证另行运行。', '',
                  '## 复现', '', '```bash', 'bash scripts/repro_chunk_graph_optimized.sh', '```', '',
                  '默认不启用优化，CPU编号需对应本机；逐项环境开关和全部原始目录见CHUNK_PIPELINE_AUTONOMOUS.md。',
                  'results.json保留阶段命令、环境、失败、指标、配对bootstrap和终态；source-sha256.txt记录交付代码。'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    paths = [path for folder in ('openwam', 'scripts', 'benchmarks/libero', 'configs', 'tests')
             for path in (ROOT / folder).rglob('*') if path.suffix in ('.py', '.sh', '.yaml')]
    (output / 'source-sha256.txt').write_text(''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}\n' for path in sorted(paths)))


if __name__ == '__main__':
    main()
