#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f outputs/robotwin/environment.sh ]]; then source outputs/robotwin/environment.sh; fi
PYTHON="${SERVER_PYTHON:-$ROOT/.venv-openwam/bin/python}"
REPEAT_ROOT="${REPEAT_ROOT:-$ROOT/outputs/robotwin/joint_quick_5h_v1}"
SESSION="${SESSION:-joint_report_3h}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-32}"
MODE="${1:-tmux}"
case "$MODE" in
    summary)
        exec "$PYTHON" -m benchmarks.utils.joint_stage_report --root "$REPEAT_ROOT"
        ;;
    tmux|run) ;;
    *) echo "Usage: bash scripts/finish_joint_report.sh [tmux|run|summary]" >&2; exit 2 ;;
esac
if [[ "$ALLOW_SHARED_GPU" != 0 && "$ALLOW_SHARED_GPU" != 1 ]]; then
    echo 'ALLOW_SHARED_GPU must be 0 or 1' >&2
    exit 2
fi
command -v timeout >/dev/null
if [[ "$MODE" == tmux ]]; then
    command -v tmux >/dev/null
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "会话已存在，未重复启动：tmux attach -t $SESSION"
        exit 0
    fi
fi
"$PYTHON" - "$REPEAT_ROOT" "$ALLOW_SHARED_GPU" "$MIN_FREE_GPU_GIB" "$MODE" <<'PY'
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
shared = sys.argv[2] == "1"
try:
    minimum_free = int(sys.argv[3])
    if minimum_free <= 0:
        raise ValueError
except ValueError:
    raise SystemExit("MIN_FREE_GPU_GIB must be a positive integer")
suite = json.loads((root / "suite.json").read_text())
if suite["gpus"] != ["4", "5", "6", "7"] or suite["repeats"] != 2 or any(case["kind"] != "difference" for case in suite["cases"]):
    raise SystemExit("Existing suite differs from the planned GPUs/repeats/controls; refusing to change identity")
if shutil.disk_usage(root).free < 5 * 1024**3:
    raise SystemExit("Less than 5 GiB disk free; experiment not started")
for path in ([] if shared else root.glob("shard_*/replay_options.json")):
    options = json.loads(path.read_text())
    if any(options.get(key) != value for key, value in {"warmup": 8, "measure": 32, "profile_samples": 2}.items()):
        raise SystemExit(f"Existing replay parameters differ: {path}; refusing to overwrite")


def query(fields):
    try:
        return subprocess.check_output(["nvidia-smi", fields, "--format=csv,noheader,nounits"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Cannot inspect NVIDIA devices; experiment not started: {error}")


gpu_rows = query("--query-gpu=index,uuid")
devices = dict(tuple(part.strip() for part in row.split(",")) for row in gpu_rows.splitlines())
if not set(suite["gpus"]).issubset(devices):
    raise SystemExit("Required GPUs 4,5,6,7 are unavailable; experiment not started")
selected = {devices[index] for index in suite["gpus"]}
contexts = query("--query-compute-apps=gpu_uuid,pid")
busy = [row for row in contexts.splitlines() if row.split(",")[0].strip() in selected]
if busy and not shared:
    raise SystemExit("Selected GPUs are busy; no processes stopped: " + "; ".join(busy))
if shared:
    free_rows = query("--query-gpu=index,memory.free")
    available = dict(tuple(part.strip() for part in row.split(",")) for row in free_rows.splitlines())
    for index in suite["gpus"]:
        try:
            free_mib = int(available[index])
        except (KeyError, ValueError):
            raise SystemExit(f"Cannot determine free memory on GPU {index}; experiment not started")
        if free_mib < minimum_free * 1024:
            raise SystemExit(f"GPU {index} has only {free_mib} MiB free; requires {minimum_free} GiB; experiment not started")
    print("WARNING: shared GPU mode; quality evaluation only; exclusive timing replay skipped. "
          "Free-memory check is not a reservation or an OOM guarantee.", flush=True)
    if sys.argv[4] == "run":
        directory = root / "reporting"
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "shared_gpu.json"
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"selected_at": datetime.now().astimezone().isoformat(),
                                        "mode": "shared_quality_only", "gpus": suite["gpus"],
                                        "minimum_free_gib": minimum_free, "foreign_contexts": busy,
                                        "free_mib": {index: int(available[index]) for index in suite["gpus"]},
                                        "timing_replay": "skipped_requires_exclusive_gpu"}, indent=2) + "\n")
        temporary.replace(marker)
print("GPU/disk preflight passed; model/kernel validation remains not_run", flush=True)
PY
if [[ "$MODE" == tmux ]]; then
    printf -v COMMAND '%q ' env "REPEAT_ROOT=$REPEAT_ROOT" "SERVER_PYTHON=$PYTHON" \
        "ALLOW_SHARED_GPU=$ALLOW_SHARED_GPU" "MIN_FREE_GPU_GIB=$MIN_FREE_GPU_GIB" bash "$ROOT/scripts/finish_joint_report.sh" run
    tmux new-session -d -s "$SESSION" "$COMMAND; STATUS=\$?; echo 运行结束，退出码=\$STATUS；124表示三小时时限; exec bash"
    echo "已启动；查看：tmux attach -t $SESSION"
    if [[ -t 0 && -t 1 && -z "${TMUX:-}" ]]; then exec tmux attach -t "$SESSION"; fi
    exit 0
fi
mkdir -p "$REPEAT_ROOT/reporting"
LOG="$REPEAT_ROOT/reporting/finish_$(date +%Y%m%d_%H%M%S).log"
PHASE=all
if [[ "$ALLOW_SHARED_GPU" == 1 ]]; then
    PHASE=run
    echo "共享 GPU：仅续跑闭环正确率，跳过独占测速；不会终止其他进程。"
fi
echo "三小时限时实验，phase=$PHASE；日志：$LOG"
STATUS=0
GPUS=4,5,6,7 timeout --signal=TERM --kill-after=60s 180m \
    bash scripts/validate_joint_repeatability.sh "$PHASE" --repeats 2 --controls 0 \
    --warmup 8 --measure 32 --profile-samples 2 --output "$REPEAT_ROOT" > "$LOG" 2>&1 || STATUS=$?
"$PYTHON" -m benchmarks.utils.joint_stage_report --root "$REPEAT_ROOT"
echo "运行结束，退出码=$STATUS；124表示三小时时限；未完成回合不计为失败。"
exit "$STATUS"
