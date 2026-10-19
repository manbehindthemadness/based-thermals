"""
Provision optional Taichi outside uv's project environment, including local ARM wheels.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from . import RETRY_SECONDS, cache_root, read_state, write_state
from .install import find_uv

LOG = logging.getLogger(__name__)
VERSION = "1.7.4"
PATH_VARIABLE = "BASED_THERMALS_TAICHI_PATH"
ARCHES = {"cpu", "vulkan", "metal", "cuda"}


def supported_python() -> bool:
    """
    Keep unsupported Python ABIs out of automatic package resolution.
    """
    supported = (sys.implementation.name == "cpython" and (3, 12) <= sys.version_info[:2] < (3, 14)
                 and sys.platform in {"darwin", "linux"})
    return supported


def wheel_host() -> bool:
    """
    Use published wheels on Apple Silicon macOS and Linux x86-64 only.
    """
    machine = platform.machine().lower()
    supported = supported_python() and (
        sys.platform == "darwin" and machine in {"arm64", "aarch64"}
        or sys.platform == "linux" and machine in {"x86_64", "amd64"}
    )
    return supported


def root_path() -> Path:
    """
    Isolate native builds by package version, interpreter ABI, platform and architecture.
    """
    root = cache_root() / (
        f"taichi-v1-{VERSION}-{sys.implementation.cache_tag}-{sys.platform}-{platform.machine()}"
    )
    return root


def environment(path: Path | None = None) -> dict[str, str]:
    """
    Select Taichi without discarding the independently selected ONNX Runtime.
    """
    env = dict(os.environ)
    previous = env.pop(PATH_VARIABLE, None)
    entries = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != previous]
    if path is not None:
        entries.insert(0, str(path))
        env[PATH_VARIABLE] = str(path)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def activate(path: Path) -> None:
    """
    Pass the cache to native imports, subprocesses and spawned workers.
    """
    previous = os.environ.get(PATH_VARIABLE)
    if previous and previous in sys.path:
        sys.path.remove(previous)
    selected = str(path)
    if selected in sys.path:
        sys.path.remove(selected)
    sys.path.insert(0, selected)
    os.environ.update({key: value for key, value in environment(path).items()
                       if key in {PATH_VARIABLE, "PYTHONPATH"}})


def query(path: Path, arch: str) -> dict[str, Any]:
    """
    Keep native crashes, teardown noise and timeouts outside the viewer process.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="based-thermals-taichi-probe-") as directory:
            response = Path(directory) / "result.json"
            completed = subprocess.run(
                [sys.executable, "-m", "based_thermals.runtime_setup.taichi_probe", arch, str(response)],
                env=environment(path), capture_output=True, text=True, check=False, timeout=45,
            )
            if completed.returncode:
                raise ValueError(f"Taichi probe exited {completed.returncode}: {completed.stderr[-2000:]}")
            result = json.loads(response.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or type(result.get("available")) is not bool:
            raise ValueError("Malformed Taichi probe response")
        if result["available"]:
            if result.get("version") != VERSION or result.get("arch") != arch:
                raise ValueError("Taichi probe version/backend mismatch")
            module = Path(result.get("module", "")).resolve()
            if not module.is_relative_to(path.resolve()):
                raise ValueError("Taichi probe imported outside the staged runtime")
            if arch == "vulkan":
                devices = re.findall(r'Vulkan Device "([^"]+)" supports Vulkan',
                                     completed.stdout + completed.stderr)
                if not devices or any(word in devices[-1].lower() for word in (
                    "llvmpipe", "lavapipe", "swiftshader", "software",
                )):
                    raise ValueError("Hardware Vulkan device not proven")
                result["device"] = devices[-1]
        return result
    except (OSError, TypeError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}


def install(path: Path, wheel: Path | None) -> bool:
    """
    Install wheels and Python helpers without shadowing the project's NumPy or native packages.
    """
    uv = find_uv()
    if uv is None:
        LOG.warning("Optional Taichi setup needs uv on PATH or beside the interpreter")
        return False
    base = [uv, "pip", "install", "--python", sys.executable, "--target", str(path),
            "--only-binary", ":all:"]
    # Taichi only needs NumPy (provided by the application) and these Python helpers.
    # Resolve helper dependencies separately so no second NumPy enters the cache.
    commands = [
        [*base, "--no-deps", str(wheel) if wheel else f"taichi=={VERSION}"],
        [*base, "colorama>=0.4.6", "dill>=0.3.8", "rich>=13"],
    ]
    try:
        for command in commands:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
            if completed.returncode:
                LOG.warning("Optional Taichi installation failed: %s", completed.stderr[-4000:])
                return False
        return True
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning("Optional Taichi installation unavailable: %s", exc)
        return False


def cached_path(root: Path, state: dict[str, Any]) -> Path | None:
    """
    Reject incomplete, symlinked or escaping cache generations.
    """
    name = state.get("ready")
    if (not isinstance(name, str) or not name.startswith("runtime-")
            or Path(name).name != name or not isinstance(state.get("arch"), str)
            or state["arch"] not in ARCHES):
        return None
    path = root / cast(str, name)
    if path.is_symlink() or not (path / "taichi/__init__.py").is_file():
        return None
    return path


def prepare(*, retry: bool = False, wheel: Path | None = None,
            arch: str | None = None, download: bool = True) -> bool:
    """
    Verify before atomic publication, preserving an existing generation on failed replacement.
    """
    root = root_path()
    verification_arch = "cpu" if arch is None else arch
    if not root.exists() and wheel is None and (not download or not wheel_host()):
        return False
    root.mkdir(parents=True, exist_ok=True)
    with (root / "setup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("Optional Taichi setup already running; using OpenCV for this launch")
            return False
        for abandoned in root.glob("staging-*"):
            if abandoned.is_dir() and not abandoned.is_symlink():
                shutil.rmtree(abandoned, ignore_errors=True)
        state_file = root / "state.json"
        state = read_state(state_file)
        cached = cached_path(root, state)
        if wheel is None and cached is not None:
            result = query(cached, verification_arch)
            if (not result.get("available") and arch is None
                    and state["arch"] != verification_arch):
                result = query(cached, state["arch"])
            if result.get("available"):
                activate(cached)
                return True
        # Linux ARM startup never downloads or compiles; only explicit local wheels can create a cache.
        if wheel is None and (not download or not wheel_host()):
            return False
        failed = state.get("failed_at")
        if (wheel is None and not retry and isinstance(failed, (int, float))
                and not isinstance(failed, bool)
                and 0 <= time.time() - failed < RETRY_SECONDS):
            return False
        stage = Path(tempfile.mkdtemp(prefix="staging-", dir=root))
        published = False
        try:
            LOG.info("Setting up optional Taichi %s", VERSION)
            if not install(stage, wheel):
                return False
            result = query(stage, verification_arch)
            if not result.get("available"):
                LOG.warning("Optional Taichi verification failed: %s", result.get("reason"))
                return False
            provenance = {"arch": verification_arch, "version": VERSION}
            if wheel:
                provenance.update(wheel=str(wheel), sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
            else:
                provenance["requirement"] = f"taichi=={VERSION}"
            committed = root / stage.name.replace("staging-", "runtime-", 1)
            stage.rename(committed)
            stage = committed
            write_state(state_file, {**provenance, "ready": stage.name})
            activate(stage)
            published = True
            LOG.info("Optional Taichi ready: %s", stage)
            return True
        finally:
            if not published:
                shutil.rmtree(stage, ignore_errors=True)
                write_state(state_file, {**state, "failed_at": time.time()})


def ensure_runtime(*, retry: bool = False, wheel: Path | None = None,
                   arch: str | None = None, download: bool = True) -> bool:
    """
    Activate a verified cache before Custom workers; unsupported/offline hosts retain OpenCV.
    """
    if (os.environ.get("BASED_THERMALS_RUNTIME_SETUP") == "0"
            or os.environ.get("BASED_THERMALS_TAICHI_SETUP") == "0" or not supported_python()):
        return False
    if "taichi" in sys.modules:
        LOG.info("Taichi already imported; restart before selecting a cached runtime")
        return False
    try:
        available = prepare(retry=retry, wheel=wheel, arch=arch, download=download)
        return available
    except (OSError, ValueError) as exc:
        LOG.warning("Optional Taichi setup unavailable: %s", exc)
        return False


def main() -> int:
    """
    Install or register a wheel without opening a camera or preparing ONNX Runtime.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry", action="store_true", help="Ignore the failed-download cooldown")
    parser.add_argument("--wheel", type=Path, help="Register an optional locally compiled wheel")
    parser.add_argument("--arch", choices=sorted(ARCHES), default="cpu", help="Backend to verify")
    parser.add_argument("--output", type=Path, help="Write selected cache path as JSON")
    args = parser.parse_args()
    if args.wheel and (not args.wheel.is_file() or args.wheel.suffix != ".whl"):
        parser.error("--wheel must name an existing wheel file")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    available = ensure_runtime(retry=args.retry, wheel=args.wheel.resolve() if args.wheel else None,
                               arch=args.arch)
    result = {"available": available, "path": os.environ.get(PATH_VARIABLE) if available else None}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    LOG.info("Optional Taichi: %s", result["path"] or "unavailable; using OpenCV")
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
