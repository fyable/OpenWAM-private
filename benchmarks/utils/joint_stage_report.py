"""Report Joint quality and replay timings without modifying hashed experiment sources."""

import argparse
from datetime import datetime
from pathlib import Path

from benchmarks.robotwin import joint_repeatability as repeat


def collect(root):
    root = Path(root).resolve()
    suite = repeat.base.read_json(root / "suite.json")
    rows = []
    for case in suite["cases"]:
        outcomes = {}
        for backend in repeat.METHODS:
            outcomes[backend] = []
            for index in range(suite["repeats"]):
                result = repeat.valid_run(root, suite, case, backend, index)
                outcomes[backend].append(bool(result["episodes"][0]["success"]) if result else None)
        rows.append({"case": case["id"], "task": case["task"], "outcomes": outcomes})
    totals = {}
    for backend in repeat.METHODS:
        values = [value for row in rows for value in row["outcomes"][backend] if value is not None]
        totals[backend] = {"completed": len(values), "successes": sum(values)}
    common = [values for row in rows for values in zip(*(row["outcomes"][name] for name in repeat.METHODS))
              if all(value is not None for value in values)]
    paired = {}
    for backend in ("prepared", "prefix"):
        pairs = [(left, right) for row in rows for left, right in
                 zip(row["outcomes"]["legacy"], row["outcomes"][backend])
                 if left is not None and right is not None]
        paired[backend] = {
            "pairs": len(pairs), "legacy_successes": sum(left for left, right in pairs),
            "successes": sum(right for left, right in pairs),
            "regressions": sum(left and not right for left, right in pairs),
            "improvements": sum(not left and right for left, right in pairs),
            "repeatable_regressions": [row["case"] for row in rows
                                       if all(value is True for value in row["outcomes"]["legacy"])
                                       and all(value is False for value in row["outcomes"][backend])],
        }
    baseline = [row["outcomes"]["legacy"] for row in rows if None not in row["outcomes"]["legacy"]]
    source = Path(suite["source"])
    full = {}
    for shard in sorted({case["shard"] for case in suite["cases"]}):
        path = source / f"shard_{shard}" / "comparison.json"
        full[str(shard)] = repeat.base.read_json(path)
    replay = {}
    for shard in range(len(suite["gpus"])):
        path = root / f"shard_{shard}" / "replay" / "timing.json"
        if path.exists():
            measured = repeat.base.read_json(path)
            if measured.get("trace", {}).get("suite_identity") != suite["identity"]:
                raise ValueError(f"Replay suite identity mismatch: {path}")
            replay[str(shard)] = measured
    expected = len(rows) * suite["repeats"]
    sharing_path = root / "reporting" / "shared_gpu.json"
    return {"generated_at": datetime.now().astimezone().isoformat(), "root": str(root),
            "suite_identity": suite["identity"], "source": str(source), "cases": len(rows),
            "tasks": len({row["task"] for row in rows}), "expected_per_backend": expected,
            "status": "complete" if all(value["completed"] == expected for value in totals.values()) else "incomplete",
            "totals": totals, "common": {"count": len(common), "successes": {
                backend: sum(value[index] for value in common) for index, backend in enumerate(repeat.METHODS)}},
            "paired": paired, "baseline_repeat": {"complete_cases": len(baseline),
                "variable_cases": sum(len(set(values)) > 1 for values in baseline)},
            "full": full, "replay": replay,
            "gpu_sharing": repeat.base.read_json(sharing_path) if sharing_path.exists() else None,
            "boundaries": "Selected-case diagnostics are not benchmark success rates or proof of equivalence. "
                          "Legacy is already Joint stage-parallel, not Native N10. Future-video latents included; VAE decode excluded."}


def rate(successes, count):
    return f"{100 * successes / count:.1f}%" if count else "未完成"


def render(report):
    lines = ["# Joint 加速阶段对比汇报", "", f"更新时间：{report['generated_at']}", "",
             "legacy 已是 Joint 阶段并行后端，不是原版 Native N10。包含未来视频 latent 去噪，不包含视频 VAE 解码。"]
    if report.get("gpu_sharing"):
        lines += ["", "本轮已选择共享 GPU 续跑：只补闭环正确率，跳过独占测速。共享运行的闭环时延受其他进程影响，"
                  "不用于正式性能对比；如有回放结果，来自另行执行的独占回放。"]
    lines += ["", "## 完整来源实验", "", "分片闭环热请求时延，不是固定输入测速；不合并或平均分片 P50。", "",
             "| 后端 | 完成/预期 | 成功回合 | 成功率 | 各分片热请求 P50 范围 | 分片内时延降低范围 |",
             "|---|---:|---:|---:|---:|---:|"]
    for backend, method in repeat.METHODS.items():
        values = [shard["methods"][method] for shard in report["full"].values()]
        episodes = sum(value["episodes"] for value in values)
        expected = sum(value["expected_episodes"] for value in values)
        successes = sum(value["successes"] for value in values)
        medians = [value["joint_request_timing"]["warm"]["p50_ms"] for value in values]
        reductions = [100 * (1 - value["joint_request_timing"]["warm"]["p50_ms"] /
                                 shard["methods"][repeat.METHODS["legacy"]]["joint_request_timing"]["warm"]["p50_ms"])
                      for value, shard in zip(values, report["full"].values())]
        lines.append(f"| {backend} | {episodes}/{expected} | {successes} | {rate(successes, episodes)} | "
                     f"{min(medians):.2f}–{max(medians):.2f} ms | {min(reductions):.1f}%–{max(reductions):.1f}% |")
    lines += ["", f"## 差异场景重复实验：{report['status']}", "",
              f"{report['tasks']} 个任务、{report['cases']} 个场景；这是差异筛选集，不能外推整体成功率。",
              "未完成回合不计为失败；完成数量不同，不直接比较各自原始成功率。", "",
              "| 后端 | 有效完成/预期 | 成功回合 | 三者共同样本成功回合 | 共同样本成功率 |",
              "|---|---:|---:|---:|---:|"]
    for backend, totals in report["totals"].items():
        count = report["common"]["count"]
        successes = report["common"]["successes"][backend]
        lines.append(f"| {backend} | {totals['completed']}/{report['expected_per_backend']} | {totals['successes']} | "
                     f"{successes}/{count} | {rate(successes, count)} |")
    lines += ["", "### 与 legacy 的逐场景配对", "", "两行配对集合可能不同，不可互相直接排序。", "",
              "| 候选 | 有效配对数 | legacy 成功 | 候选成功 | 退化 | 改善 |", "|---|---:|---:|---:|---:|---:|"]
    for backend, value in report["paired"].items():
        lines.append(f"| {backend} | {value['pairs']} | {value['legacy_successes']} | {value['successes']} | "
                     f"{value['regressions']} | {value['improvements']} |")
    baseline = report["baseline_repeat"]
    lines += ["", f"legacy 完整重复场景中，{baseline['variable_cases']}/{baseline['complete_cases']} 个出现结果波动；"
              "不能用基线波动自动解释候选退化。", "", "### 所有重复均发生退化的场景", ""]
    for backend, value in report["paired"].items():
        lines.append(f"- {backend}：" + (", ".join(value["repeatable_regressions"]) or "当前未观察到（不代表证明无损）"))
    lines += ["", "## 本轮固定输入回放", "", "验证状态与速度分开报告；parity 失败时仅为诊断测速。",
              "未生成的分片标记为未完成，不引用旧实验时延冒充本轮结果。", "",
              "| 分片 | 后端 | 计时样本 | P50 ms | P95 ms | 相对 legacy 加速比 | 验证解释 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for shard in report["full"]:
        measured = report["replay"].get(shard)
        if measured is None or measured.get("status") != "complete":
            lines.append(f"| {shard} | — | — | — | — | — | 未完成 |")
            continue
        for backend, value in measured["methods"].items():
            wall = value["wall_ms"]
            speedup = measured["methods"]["legacy"]["wall_ms"]["p50"] / wall["p50"]
            lines.append(f"| {shard} | {backend} | {wall['count']} | {wall['p50']:.2f} | {wall['p95']:.2f} | "
                         f"{speedup:.3f}x | {measured['interpretation']} |")
    lines += ["", "## 结论边界", "", "- prepared 是优先验证候选；总体成功率持平不等于逐场景无损。",
              "- prefix 保留为实验候选；稳定退化、预测误差和长尾未解释前不升级默认。",
              "- 四卡只用于评测分片，不代表单请求多卡加速；固定输入回放不验证闭环成功率。",
              "- 已知 parity 失败的阈值不放宽，不宣称数值等价或统计非劣性。", "",
              f"来源完整实验：`{report['source']}`", f"本轮重复实验：`{report['root']}`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repeat.base.ROOT / "outputs/robotwin/joint_quick_5h_v1")
    args = parser.parse_args()
    report = collect(args.root)
    directory = args.root / "reporting"
    repeat.base.write_json(directory / "comparison.json", report)
    temporary = directory / "comparison.md.tmp"
    temporary.write_text(render(report))
    temporary.replace(directory / "comparison.md")
    print(directory / "comparison.md")


if __name__ == "__main__":
    main()
