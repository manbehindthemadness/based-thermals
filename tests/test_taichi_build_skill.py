"""Verify source-build recovery and required CPU fallback without native compilation."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".agents/skills/build-taichi/scripts/build_taichi.py"
spec = importlib.util.spec_from_file_location("taichi_build_helper", SCRIPT)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.mark.parametrize("gpu", ["hardware", "software", "unavailable", "teardown_crash"])
def test_gpu_failure_keeps_verified_cpu(monkeypatch, tmp_path, gpu):
    def run(command, *, log, timeout):
        arch = command[command.index("--arch") + 1]
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"ok": True, "actual_arch": arch}))
        device = "llvmpipe" if gpu == "software" else "NVIDIA GB10"
        log.write_text(f'Vulkan Device "{device}" supports Vulkan 0 version 1.4.312')
        if arch == "vulkan" and gpu in {"unavailable", "teardown_crash"}:
            if gpu == "unavailable":
                output.unlink()
            raise subprocess.CalledProcessError(-11, command)

    monkeypatch.setattr(build, "run", run)
    verified, device = build.verify_backends(Path("/fake/python"), tmp_path)
    assert "cpu" in verified
    assert ("vulkan" in verified) is (gpu == "hardware")
    assert device == ("NVIDIA GB10" if gpu == "hardware" else None)


def test_failed_cpu_stops_before_gpu_or_registration(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command[command.index("--arch") + 1])
        raise subprocess.CalledProcessError(-7, command)

    monkeypatch.setattr(build, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        build.verify_backends(Path("/fake/python"), tmp_path)
    assert calls == ["cpu"]


def test_arm_atomic_patch_is_repeatable_and_refuses_unrelated_edits(tmp_path):
    """Exercise actual patch application against its pinned context, without any network."""
    patch = SCRIPT.parents[1] / "assets/arm64-no-outline-atomics.patch"
    # Reconstruct the complete hunk context needed to apply the patch locally.
    lines = patch.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("@@"))
    original = "\n".join(line[1:] for line in lines[start + 1:]
                         if line.startswith((" ", "-"))) + "\n"
    target = tmp_path / "taichi/runtime/llvm/runtime_module/CMakeLists.txt"
    target.parent.mkdir(parents=True)
    target.write_text("# ./taichi/runtime/llvm/runtime_module/CMakeLists.txt\n" + original)
    for command in (["git", "init", "-q"], ["git", "add", "."],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                     "commit", "-qm", "fixture"]):
        subprocess.run(command, cwd=tmp_path, check=True)
    first = build.patch_source(tmp_path)
    assert "-mno-outline-atomics" in target.read_text()
    assert build.patch_source(tmp_path) == first
    target.write_text(target.read_text() + "# unrelated edit\n")
    with pytest.raises(RuntimeError, match="unrelated tracked edits"):
        build.patch_source(tmp_path)
