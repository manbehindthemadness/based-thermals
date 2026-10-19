"""
Exercise optional Taichi provisioning without live installers or native imports.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from based_thermals import runtime_setup
from based_thermals.runtime_setup import taichi as setup


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """
    Keep downloads, cache writes and native probes inside fake boundaries.
    """
    monkeypatch.delenv("BASED_THERMALS_RUNTIME_SETUP", raising=False)
    monkeypatch.delenv("BASED_THERMALS_TAICHI_SETUP", raising=False)
    monkeypatch.delitem(setup.sys.modules, "taichi", raising=False)
    monkeypatch.setattr(setup, "supported_python", lambda: True)
    monkeypatch.setattr(setup, "wheel_host", lambda: True)
    root = tmp_path / "cache"
    monkeypatch.setattr(setup, "root_path", lambda: root)

    def install(path: Path, _wheel: Path | None) -> bool:
        """
        Materialize a fake staged package without executing native code.
        """
        (path / "taichi").mkdir()
        (path / "taichi/__init__.py").write_text("# fake staged native package\n")
        return True

    install_mock = Mock(side_effect=install)
    query = Mock(return_value={"available": True})
    activate = Mock()
    monkeypatch.setattr(setup, "install", install_mock)
    monkeypatch.setattr(setup, "query", query)
    monkeypatch.setattr(setup, "activate", activate)
    return SimpleNamespace(root=root, install=install_mock, query=query, activate=activate)


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("darwin", "arm64", True),
        ("darwin", "x86_64", False),
        ("linux", "x86_64", True),
        ("linux", "aarch64", False),
        ("linux", "arm64", False),
        ("win32", "AMD64", False),
    ],
)
def test_wheel_hosts(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, expected: bool
) -> None:
    """
    Limit automatic downloads to published wheel platforms.
    """
    monkeypatch.setattr(setup.sys, "platform", system)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    assert setup.wheel_host() is expected


@pytest.mark.parametrize("version", [(3, 11), (3, 14), (3, 15)])
def test_unsupported_python_does_not_resolve_wheels(
    monkeypatch: pytest.MonkeyPatch, version: tuple[int, int]
) -> None:
    """
    Skip wheel resolution on unsupported Python ABIs.
    """
    monkeypatch.setattr(setup.sys, "version_info", version)
    assert not setup.wheel_host()


def test_publish_and_reuse_without_reinstall(isolated: SimpleNamespace) -> None:
    """
    Publish a verified generation once and reuse it without downloading.
    """
    assert setup.ensure_runtime()
    state = json.loads((isolated.root / "state.json").read_text())
    selected = isolated.root / state["ready"]
    assert state["requirement"] == "taichi==1.7.4"
    assert not list(isolated.root.glob("staging-*"))
    assert selected.is_dir()
    assert setup.ensure_runtime()
    isolated.install.assert_called_once()
    assert isolated.query.call_args.args == (selected, "cpu")
    isolated.activate.assert_called_with(selected)


def test_arm_only_uses_explicitly_registered_wheel(
    monkeypatch: pytest.MonkeyPatch, isolated: SimpleNamespace, tmp_path: Path
) -> None:
    """
    Require explicit local registration before ARM cache activation.
    """
    monkeypatch.setattr(setup, "wheel_host", lambda: False)
    assert not setup.ensure_runtime()
    assert not isolated.root.exists()
    isolated.install.assert_not_called()
    wheel = tmp_path / "local.whl"
    wheel.write_bytes(b"local build")
    assert setup.ensure_runtime(wheel=wheel, arch="vulkan")
    state = json.loads((isolated.root / "state.json").read_text())
    assert state["arch"] == "vulkan" and len(state["sha256"]) == 64
    assert setup.ensure_runtime(arch="vulkan")
    isolated.install.assert_called_once()
    assert isolated.query.call_args.args[1] == "vulkan"


def test_cached_runtime_verifies_requested_backend(isolated: SimpleNamespace) -> None:
    """
    A CPU-verified cache cannot falsely satisfy an explicit Vulkan verification request.
    """
    assert setup.ensure_runtime()
    isolated.query.side_effect = lambda _path, arch: {"available": arch == "cpu"}
    isolated.activate.reset_mock()
    assert not setup.ensure_runtime(arch="vulkan", download=False)
    assert isolated.query.call_args.args[1] == "vulkan"
    isolated.activate.assert_not_called()
    isolated.install.assert_called_once()
    assert setup.ensure_runtime(download=False)


def test_default_activation_does_not_require_cached_gpu(
    isolated: SimpleNamespace,
    tmp_path: Path,
) -> None:
    """
    Recheck CPU on a GPU-registered build so driver failure does not hide a working CPU.
    """
    wheel = tmp_path / "local.whl"
    wheel.write_bytes(b"CPU and Vulkan build")
    assert setup.ensure_runtime(wheel=wheel, arch="vulkan")
    isolated.query.side_effect = lambda _path, arch: {"available": arch == "cpu"}
    assert setup.ensure_runtime(download=False)
    assert isolated.query.call_args.args[1] == "cpu"


def test_legacy_gpu_cache_remains_usable_without_claiming_cpu_support(
    isolated: SimpleNamespace, tmp_path: Path,
) -> None:
    """
    Preserve automatic legacy GPU activation while requiring proof for explicit CPU requests.
    """
    wheel = tmp_path / "legacy.whl"
    wheel.write_bytes(b"Vulkan-only build")
    assert setup.ensure_runtime(wheel=wheel, arch="vulkan")
    isolated.query.reset_mock()
    isolated.query.side_effect = lambda _path, arch: {"available": arch == "vulkan"}
    assert setup.ensure_runtime(download=False)
    assert [call.args[1] for call in isolated.query.call_args_list] == ["cpu", "vulkan"]
    isolated.activate.reset_mock()
    assert not setup.ensure_runtime(arch="cpu", download=False)
    isolated.activate.assert_not_called()


def test_experiment_cache_activation_never_downloads(isolated: SimpleNamespace) -> None:
    """
    Allow experiment cache reuse without any network installation.
    """
    assert not setup.ensure_runtime(download=False)
    isolated.install.assert_not_called()
    assert setup.ensure_runtime()
    assert setup.ensure_runtime(download=False)
    isolated.install.assert_called_once()


@pytest.mark.parametrize("switch", ["BASED_THERMALS_RUNTIME_SETUP", "BASED_THERMALS_TAICHI_SETUP"])
def test_disable_prevents_discovery_and_cache_writes(
    monkeypatch: pytest.MonkeyPatch, isolated: SimpleNamespace, switch: str
) -> None:
    """
    Honor setup switches before discovery or cache mutation.
    """
    monkeypatch.setenv(switch, "0")
    assert not setup.ensure_runtime()
    assert not isolated.root.exists()
    isolated.install.assert_not_called()
    isolated.query.assert_not_called()


def test_imported_runtime_is_never_replaced(
    monkeypatch: pytest.MonkeyPatch, isolated: SimpleNamespace
) -> None:
    """
    Leave already imported native modules intact.
    """
    monkeypatch.setitem(setup.sys.modules, "taichi", object())
    assert not setup.ensure_runtime()
    isolated.query.assert_not_called()
    isolated.install.assert_not_called()


@pytest.mark.parametrize("failure", ["install", "probe"])
def test_failed_stage_cleanup_cooldown_and_retry(
    monkeypatch: pytest.MonkeyPatch, isolated: SimpleNamespace, failure: str
) -> None:
    """
    Clean failed stages and honor retry cooldowns.
    """
    if failure == "install":
        isolated.install.side_effect = None
        isolated.install.return_value = False
    else:
        isolated.query.return_value = {"available": False, "reason": "native failure"}
    assert not setup.ensure_runtime()
    assert not list(isolated.root.glob("staging-*"))
    assert not list(isolated.root.glob("runtime-*"))
    assert "failed_at" in json.loads((isolated.root / "state.json").read_text())
    assert not setup.ensure_runtime()
    isolated.install.assert_called_once()
    assert not setup.ensure_runtime(retry=True)
    assert isolated.install.call_count == 2
    isolated.activate.assert_not_called()


def test_failed_replacement_retains_successful_generation(
    isolated: SimpleNamespace, tmp_path: Path
) -> None:
    """
    Retain the previous successful generation after failed replacement.
    """
    assert setup.ensure_runtime()
    previous = json.loads((isolated.root / "state.json").read_text())
    wheel = tmp_path / "broken.whl"
    wheel.write_bytes(b"broken")
    isolated.query.return_value = {"available": False, "reason": "crash"}
    assert not setup.ensure_runtime(wheel=wheel, arch="vulkan")
    state = json.loads((isolated.root / "state.json").read_text())
    assert state["ready"] == previous["ready"]
    assert (isolated.root / previous["ready"]).is_dir()
    isolated.query.return_value = {"available": True}
    assert setup.ensure_runtime()
    assert isolated.install.call_count == 2


def test_interrupted_staging_is_removed(isolated: SimpleNamespace) -> None:
    """
    Remove interrupted installs before staging a fresh generation.
    """
    abandoned = isolated.root / "staging-interrupted"
    abandoned.mkdir(parents=True)
    (abandoned / "incomplete").write_text("partial")
    assert setup.ensure_runtime()
    assert not abandoned.exists()


def test_concurrent_setup_does_not_install(
    monkeypatch: pytest.MonkeyPatch, isolated: SimpleNamespace
) -> None:
    """
    Skip installation while another process holds the setup lock.
    """
    monkeypatch.setattr(setup.fcntl, "flock", Mock(side_effect=BlockingIOError))
    assert not setup.ensure_runtime()
    isolated.install.assert_not_called()


def test_read_only_cache_keeps_optional_fallback(
    monkeypatch: pytest.MonkeyPatch, isolated: SimpleNamespace
) -> None:
    """
    Keep OpenCV available when the runtime cache is read-only.
    """
    monkeypatch.setattr(setup, "prepare", Mock(side_effect=PermissionError("read only")))
    assert not setup.ensure_runtime()


@pytest.mark.parametrize("name", ["../escape", "runtime-../../escape", "other-package", None])
def test_cache_state_cannot_escape_root(tmp_path: Path, name: Any) -> None:
    """
    Reject generation names that escape the application-owned cache root.
    """
    assert setup.cached_path(tmp_path, {"ready": name, "arch": "cpu"}) is None


def test_cache_symlink_is_rejected(tmp_path: Path) -> None:
    """
    Refuse cache generations redirected through symlinks.
    """
    outside = tmp_path / "outside/taichi"
    outside.mkdir(parents=True)
    (outside / "__init__.py").write_text("fake")
    (tmp_path / "runtime-link").symlink_to(outside.parent, target_is_directory=True)
    assert setup.cached_path(tmp_path, {"ready": "runtime-link", "arch": "cpu"}) is None


@pytest.mark.parametrize("arch", [[], {}, None, 1, "unknown"])
def test_malformed_backend_state_is_rejected(tmp_path: Path, arch: Any) -> None:
    """
    Reject invalid backend labels before native discovery.
    """
    assert setup.cached_path(tmp_path, {"ready": "runtime-old", "arch": arch}) is None


def test_activation_preserves_onnx_and_propagates_to_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Keep independent ONNX selection while propagating Taichi to child workers.
    """
    monkeypatch.setattr(setup.sys, "path", ["/old/taichi", "/ort", "/project"])
    monkeypatch.setenv(setup.PATH_VARIABLE, "/old/taichi")
    monkeypatch.setenv("BASED_THERMALS_ORT_PATH", "/ort")
    monkeypatch.setenv("PYTHONPATH", "/old/taichi:/ort:/project")
    setup.activate(tmp_path)
    assert setup.sys.path == [str(tmp_path), "/ort", "/project"]
    assert setup.os.environ["PYTHONPATH"] == f"{tmp_path}:/ort:/project"
    assert setup.os.environ["BASED_THERMALS_ORT_PATH"] == "/ort"
    assert setup.environment()["PYTHONPATH"] == "/ort:/project"
    assert (
        runtime_setup.environment(Path("/new-ort"))["PYTHONPATH"] == f"/new-ort:{tmp_path}:/project"
    )


def test_installer_keeps_numpy_in_project_and_requires_binary_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Install binary helpers without a second NumPy in the native cache.
    """
    monkeypatch.setitem(vars(setup), "find_uv", lambda: "/fake/uv")
    execute = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(setup.subprocess, "run", execute)
    assert setup.install(tmp_path, None)
    commands = [call.args[0] for call in execute.call_args_list]
    assert "--no-deps" in commands[0] and commands[0][-1] == "taichi==1.7.4"
    assert "--no-deps" not in commands[1]
    assert all("--target" in cmd and "--only-binary" in cmd for cmd in commands)
    assert all(not arg.startswith("numpy") for cmd in commands for arg in cmd)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stderr="no wheel"),
        OSError("offline"),
        subprocess.TimeoutExpired("uv", 600),
    ],
)
def test_installation_errors_do_not_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: Any
) -> None:
    """
    Convert installer exits, I/O errors and timeouts into optional fallback.
    """
    monkeypatch.setitem(vars(setup), "find_uv", lambda: "/fake/uv")
    monkeypatch.setattr(setup.subprocess, "run", Mock(side_effect=[result]))
    assert not setup.install(tmp_path, None)


@pytest.mark.parametrize("mode", ["valid", "crash", "timeout", "malformed", "outside", "software"])
def test_native_probe_containment_and_device_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    """
    Reject crashes, malformed replies, external imports and software Vulkan.
    """

    def execute(command: list[Any], **_kwargs: Any) -> SimpleNamespace:
        """
        Return a controlled native probe reply without spawning a process.
        """
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 45)
        response: dict[str, Any] = {
            "available": True,
            "version": "1.7.4",
            "arch": "vulkan",
            "module": str(tmp_path / "taichi/__init__.py"),
        }
        if mode == "malformed":
            response["available"] = 1
        if mode == "outside":
            response["module"] = "/venv/taichi/__init__.py"
        Path(command[-1]).write_text(json.dumps(response))
        device = "llvmpipe" if mode == "software" else "NVIDIA GB10"
        return SimpleNamespace(
            returncode=-11 if mode == "crash" else 0,
            stdout="native stdout with teardown noise",
            stderr=f'Vulkan Device "{device}" supports Vulkan 0 version 1.4.312',
        )

    monkeypatch.setattr(setup.subprocess, "run", execute)
    result = setup.query(tmp_path, "vulkan")
    assert result["available"] is (mode == "valid")
    if mode == "valid":
        assert result["device"] == "NVIDIA GB10"


def test_taichi_setup_precedes_onnx_early_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run Taichi setup even when ONNX Runtime has already been imported.
    """
    monkeypatch.delenv("BASED_THERMALS_RUNTIME_SETUP", raising=False)
    monkeypatch.setitem(setup.sys.modules, "onnxruntime", object())
    taichi = Mock()
    monkeypatch.setattr(setup, "ensure_runtime", taichi)
    runtime_setup.ensure_runtime(retry=True)
    taichi.assert_called_once_with(retry=True)
