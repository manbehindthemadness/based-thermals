"""
Keep passive memory observations restricted to the requested process family.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.watch_memory import descendants, main, parse_processes


def test_process_parser_handles_executable_spaces_and_rejects_invalid_rows() -> None:
    """
    Preserve timestamps and convert RSS KiB without confusing command path components.
    """
    output = """10 1 256 2048 Sat Oct 17 18:00:00 2026 /App Folder/Python
11 10 32 3000 Sat Oct 17 18:00:01 2026 /usr/bin/sudo
12 11 128 4096 Sat Oct 17 18:00:02 2026 /usr/bin/Python
13 1 999 999 Sat Oct 17 18:00:00 2026 unrelated
bad row
X 1 1 1 Sat Oct 17 18:00:00 2026 invalid
14 1 -1 1 Sat Oct 17 18:00:00 2026 invalid
"""
    processes = parse_processes(output)
    family = descendants(processes, 10)
    assert [item["pid"] for item in family] == [10, 11, 12]
    assert family[0]["executable"] == "/App Folder/Python"
    assert family[0]["rss_bytes"] == 256 * 1024
    assert family[0]["started"] == "Sat Oct 17 18:00:00 2026"
    assert len(processes) == 4
    assert descendants(processes, 999) == []


@pytest.mark.parametrize("replacement", [False, True])
def test_monitor_stops_on_exit_or_reused_pid_without_controlling_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: bool,
) -> None:
    """
    Write passive RSS samples and stop when the original viewer identity disappears.
    """
    first = "10 1 256 2048 Sat Oct 17 18:00:00 2026 Python"
    second = first.replace("18:00:00", "19:00:00") if replacement else ""
    run = Mock(side_effect=[SimpleNamespace(stdout=first), SimpleNamespace(stdout=second)])
    monkeypatch.setattr("tools.watch_memory.subprocess.run", run)
    monkeypatch.setattr("tools.watch_memory.time.sleep", lambda _: None)
    output = tmp_path / "memory.jsonl"
    assert main(["10", "--output", str(output)]) == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["processes"][0]["rss_bytes"] == 256 * 1024
    assert records[1]["event"] == ("pid_reused" if replacement else "process_exited")
    assert records[1]["processes"] == []
    assert all(call.args[0][0] == "ps" for call in run.call_args_list)
