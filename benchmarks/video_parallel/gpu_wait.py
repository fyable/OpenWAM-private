import argparse
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def snapshot():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    devices = {}
    for row in csv.reader(io.StringIO(result)):
        index, uuid, memory, utilization = [value.strip() for value in row]
        devices[index] = (uuid, float(memory), float(utilization))
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits",
    ], text=True)
    busy = {value.strip() for value in processes.splitlines() if value.strip()}
    return devices, busy


def available(devices, busy, memory_limit, utilization_limit):
    return {index for index, (uuid, memory, utilization) in devices.items()
            if uuid not in busy and math.isfinite(memory) and math.isfinite(utilization)
            and memory <= memory_limit and utilization <= utilization_limit}


def reserve(uuids):
    locks = []
    try:
        for uuid in sorted(uuids):
            path = Path(tempfile.gettempdir()) / f"openwam-gpu-{os.getuid()}-{uuid}.lock"
            handle = open(path, "a")
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locks
    except BlockingIOError:
        for handle in locks:
            handle.close()
        return []


def main():
    parser = argparse.ArgumentParser(description="Wait for idle physical GPUs, then exec a command.")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--poll", type=float, default=30)
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=0)
    parser.add_argument("--max-memory-mib", type=float, default=1024)
    parser.add_argument("--max-utilization", type=float, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.poll <= 0 or min(args.idle_seconds, args.timeout, args.max_memory_mib,
                                            args.max_utilization) < 0:
        parser.error("command required; poll must be positive and thresholds nonnegative")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        parser.error("unset CUDA_VISIBLE_DEVICES first; GPU indices here are physical nvidia-smi indices")
    requested = None if args.gpus == "auto" else args.gpus.split(",")
    if requested is not None and (not all(value.isdigit() for value in requested)
                                  or len(set(requested)) != len(requested)):
        parser.error("--gpus must be auto or comma-separated physical indices")
    started = time.monotonic()
    idle_since = {}
    print(f"Waiting for GPUs={args.gpus}, idle={args.idle_seconds:g}s, "
          f"memory<={args.max_memory_mib:g}MiB, utilization<={args.max_utilization:g}%", flush=True)
    while True:
        now = time.monotonic()
        if args.timeout and now - started >= args.timeout:
            raise TimeoutError("GPU wait timeout; experiment was not started")
        devices, busy = snapshot()
        if requested is not None and not set(requested) <= devices.keys():
            raise ValueError(f"Unknown GPU indices: {requested}; available: {list(devices)}")
        free = available(devices, busy, args.max_memory_mib, args.max_utilization)
        idle_since = {index: idle_since.get(index, now) for index in free}
        ready = {index for index in free if now - idle_since[index] >= args.idle_seconds}
        selected = (sorted(ready, key=int)[:1] if requested is None
                    else requested if set(requested) <= ready else [])
        if selected:
            locks = reserve([devices[index][0] for index in selected])
            if locks:
                confirmed, busy = snapshot()
                if (set(selected) <= available(confirmed, busy, args.max_memory_mib, args.max_utilization)
                        and all(confirmed[index][0] == devices[index][0] for index in selected)):
                    environment = dict(os.environ, WAIT_FOR_GPU="0", GPU=selected[0],
                                       GPUS=",".join(selected), DEVICE=f"cuda:{selected[0]}")
                    for handle in locks:
                        os.set_inheritable(handle.fileno(), True)
                    print(f"Starting on GPU {','.join(selected)}: {' '.join(command)}", flush=True)
                    os.execvpe(command[0], command, environment)
                for handle in locks:
                    handle.close()
        waiting_for = "any GPU" if requested is None else f"requested GPUs={','.join(requested)}"
        print(f"Still waiting for {waiting_for}; idle candidates: "
              f"{','.join(sorted(free, key=int)) or 'none'}; "
              f"stable candidates: {','.join(sorted(ready, key=int)) or 'none'}", flush=True)
        delay = args.poll if not args.timeout else min(args.poll, max(0, args.timeout - (time.monotonic() - started)))
        time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
