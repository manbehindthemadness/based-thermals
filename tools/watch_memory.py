"""
Sample a running viewer and its helper processes without opening USB or modifying the app.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_processes(output: str) -> dict[int, dict[str, Any]]:
    """
    Parse macOS/Linux ps rows, retaining process identity and current resident bytes.
    """
    processes = {}
    for line in output.splitlines():
        fields = line.split(None, 9)
        if len(fields) != 10:
            continue
        try:
            pid, parent, rss, virtual = (int(value) for value in fields[:4])
        except ValueError:
            continue
        if pid < 1 or min(parent, rss, virtual) < 0:
            continue
        processes[pid] = {
            "pid": pid, "parent": parent, "rss_bytes": rss * 1024,
            "virtual_bytes": virtual * 1024, "started": " ".join(fields[4:9]),
            "executable": fields[9],
        }
    return processes


def descendants(processes: dict[int, dict[str, Any]], root: int) -> list[dict[str, Any]]:
    """
    Select only the viewer and its current descendants, including inference and USB helpers.
    """
    if root not in processes:
        return []
    selected = {root}
    while True:
        children = {pid for pid, item in processes.items() if item["parent"] in selected}
        added = children - selected
        if not added:
            return [processes[pid] for pid in sorted(selected)]
        selected.update(added)


def main(argv: list[str] | None = None) -> int:
    """
    Write bounded-duration memory samples until the original viewer exits or monitoring stops.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--duration", type=float, default=4 * 3600)
    args = parser.parse_args(argv)
    if args.pid < 1 or not 1 <= args.interval <= 3600 or not 1 <= args.duration <= 86400:
        parser.error("PID must be positive; interval 1–3600 seconds; duration 1–86400 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration
    identity = None
    try:
        with args.output.open("x", buffering=1, encoding="utf-8") as output:
            while time.monotonic() < deadline:
                result = subprocess.run(
                    ["ps", "-axo", "pid=,ppid=,rss=,vsz=,lstart=,comm="],
                    capture_output=True, text=True, check=True, timeout=3,
                )
                processes = parse_processes(result.stdout)
                family = descendants(processes, args.pid)
                current = processes.get(args.pid)
                replaced = current is not None and identity is not None and (
                    current["started"] != identity
                )
                event = "pid_reused" if replaced else "sample" if current else "process_exited"
                if current is not None:
                    identity = processes[args.pid]["started"]
                record = {
                    "time": datetime.now(UTC).isoformat(), "monotonic": time.monotonic(),
                    "event": event, "processes": family if event == "sample" else [],
                }
                output.write(json.dumps(record) + "\n")
                if event != "sample":
                    return 0
                time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        return 0  # Stopping this monitor never signals or closes the viewer.
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Memory monitor failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
