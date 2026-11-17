#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f outputs/robotwin/environment.sh ]]; then
    source outputs/robotwin/environment.sh
fi
PYTHON="${SERVER_PYTHON:-$ROOT/.venv-openwam/bin/python}"
MODE="${1:-tmux}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" == tmux ]]; then
    command -v tmux >/dev/null || { echo 'tmux 未安装；可使用本脚本 run 模式。' >&2; exit 1; }
    SESSION="${SESSION:-joint_repeatability}"
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "会话 $SESSION 已存在，未重复启动。进入：tmux attach -t $SESSION"
        exit 0
    fi
    "$PYTHON" -m benchmarks.robotwin.joint_repeatability plan "$@"
    COMMAND=(env "GPUS=${GPUS:-0,1,2,3}" "BASE_PORT=${BASE_PORT:-8950}" "REPEATS=${REPEATS:-5}"
        "CONTROL_CASES=${CONTROL_CASES:-10}" "SERVER_PYTHON=$PYTHON"
        "SOURCE_ROOT=${SOURCE_ROOT:-$ROOT/outputs/robotwin/joint_three_backend_50tasks_e5_4gpu_v1}"
        "REPEAT_ROOT=${REPEAT_ROOT:-$ROOT/outputs/robotwin/joint_repeatability_v1}")
    for NAME in ROBOTWIN_PATH ROBOTWIN_PYTHON SERVER_START_TIMEOUT TORCH_EXTENSIONS_DIR WARP_CACHE_PATH XDG_CACHE_HOME MPLCONFIGDIR; do
        if [[ -n "${!NAME:-}" ]]; then COMMAND+=("$NAME=${!NAME}"); fi
    done
    COMMAND+=(bash "$ROOT/scripts/validate_joint_repeatability.sh" all "$@")
    printf -v QUOTED '%q ' "${COMMAND[@]}"
    tmux new-session -d -s "$SESSION" "$QUOTED; STATUS=\$?; echo 完成，退出码=\$STATUS; exec bash"
    echo "已启动 $SESSION；退出界面不停止：Ctrl+b 然后 d。"
    if [[ -t 0 && -t 1 && -z "${TMUX:-}" ]]; then
        exec tmux attach-session -t "$SESSION"
    fi
    echo "进入：tmux attach -t $SESSION"
    exit 0
fi
exec "$PYTHON" -m benchmarks.robotwin.joint_repeatability "$MODE" "$@"
