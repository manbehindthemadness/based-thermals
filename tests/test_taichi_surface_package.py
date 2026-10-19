"""
Verify the optional Custom package with fake Taichi boundaries, never live GPU kernels or downloads.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from support.pipeline import raw_pipeline
from support.qt_process import run_popup

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import node, validate_pipeline

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "examples/custom_nodes/Thermal Surface Taichi"


def package() -> tuple[CustomProcessor, dict[str, Any], Any]:
    """
    Load the real portable node with explicit OpenCV execution to keep tests independent of Taichi.
    """
    item = node("software", "custom", **load_folder(FOLDER))
    execution(item, "OpenCV")
    item = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))["software"][2]
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    processor.apply(np.full((24, 32, 3), 120, np.float32), item)
    module = sys.modules[processor.packages[item["id"]].name]
    return processor, item, module


def test_screenshot_defaults_match_embedded_and_standalone_settings() -> None:
    """
    Preserve every selected screenshot setting in portable metadata and missing-key fallbacks.
    """
    expected = {"execution": "Taichi CPU", "detail": 256, "yaw": 80, "tilt": 45,
                "scene_rotation": 90, "offset_x": -14, "offset_y": 0, "zoom": 1,
                "height": .7, "smoothing": 3, "palette": "Turbo",
                "normalization": "Source range", "resolution": "1280", "wireframe": False,
                "antialiasing": "2x supersampling"}
    assert json.loads(load_folder(FOLDER)["config"]) == expected
    processor, _, module = package()
    try:
        settings = sys.modules[module.__package__ + ".settings"].Settings.read({})
        actual = vars(settings).copy()
        actual["resolution"] = str(actual["resolution"])
        assert actual == expected
    finally:
        processor.close()


def execution(item: dict[str, Any], value: object) -> None:
    """
    Change only this node's current optional execution setting.
    """
    item["params"]["config"] = json.dumps({**json.loads(item["params"]["config"]), "execution": value})


@pytest.mark.parametrize("detail", [128, 256, 512])
def test_dense_mesh_survives_saved_json_and_renders(detail: int) -> None:
    """
    Preserve dense settings and full topology through the actual portable processing boundary.
    """
    processor, item, module = package()
    image = np.repeat(np.linspace(0, 255, 32, dtype=np.float32)[None, :, None], 24, axis=0)
    image = np.repeat(image, 3, axis=2)
    original = image.copy()
    try:
        config = {**json.loads(item["params"]["config"]), "detail": detail}
        item["params"]["config"] = json.dumps(config)
        restored = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))["software"][2]
        result = processor.apply(image, restored)
        assert json.loads(restored["params"]["config"])["detail"] == detail
        assert result.shape == (960, 1280, 3) and result.dtype == np.float32
        assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
        geometry = module._renderer.geometry
        assert (geometry.rows, geometry.columns) == (detail * 3 // 4, detail)
        expected_faces = 2 * (geometry.rows - 1) * (detail - 1) + 4 * (geometry.rows + detail - 2)
        assert geometry.triangles.shape == (expected_faces, 3)
        assert geometry.triangles.min() == 0
        assert geometry.triangles.max() < 2 * geometry.rows * detail
        np.testing.assert_array_equal(image, original)
    finally:
        processor.close()


@pytest.mark.parametrize("detail", [513, 11, 256.5, True])
def test_dense_mesh_rejects_invalid_settings(detail: object) -> None:
    """
    Keep the expanded mesh allocation bounded and reject malformed numeric settings.
    """
    processor, _, module = package()
    try:
        settings = sys.modules[module.__package__ + ".settings"]
        with pytest.raises(ValueError, match="detail"):
            settings.Settings.read({"detail": detail})
    finally:
        processor.close()


@pytest.mark.parametrize("shape,expected", [
    ((192, 256), (1732, 2309)),
    ((256, 192), (2309, 1732)),
    ((256, 256), (2000, 2000)),
    ((48, 256), (768, 4096)),
])
def test_large_output_preserves_aspect_within_pipeline_limit(
    shape: tuple[int, int], expected: tuple[int, int],
) -> None:
    """
    Render saved 4096 requests across input aspects without exceeding four megapixels.
    """
    processor, item, _ = package()
    try:
        config = {**json.loads(item["params"]["config"]), "resolution": "4096", "detail": 12}
        item["params"]["config"] = json.dumps(config)
        restored = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))["software"][2]
        result = processor.apply(np.full((*shape, 3), 120, np.float32), restored)
        assert result.shape == (*expected, 3)
        assert result.shape[0] * result.shape[1] <= 4_000_000
        assert abs(result.shape[1] / result.shape[0] - shape[1] / shape[0]) < .002
        assert np.isfinite(result).all()
        assert json.loads(restored["params"]["config"])["resolution"] == "4096"
        config["resolution"] = "640"
        restored["params"]["config"] = json.dumps(config)
        smaller = processor.apply(np.full((*shape, 3), 120, np.float32), restored)
        assert max(smaller.shape[:2]) == 640
    finally:
        processor.close()


def test_portable_opencv_matches_original_and_never_imports_taichi(monkeypatch: Any) -> None:
    """
    Preserve the original surface exactly without installing or probing an optional runtime.
    """
    processor, item, module = package()
    try:
        def forbidden(_execution: str) -> None:
            """
            Fail if the default path tries to initialize any native runtime.
            """
            raise AssertionError("OpenCV must not initialize Taichi")

        surface = sys.modules[module.__package__ + ".surface"]
        monkeypatch.setitem(vars(surface), "Rasterizer", forbidden)
        defaults = json.loads(item["params"]["config"])
        assert (defaults["detail"], defaults["resolution"]) == (256, "1280")
        settings = sys.modules[module.__package__ + ".settings"].Settings.read({})
        assert (settings.detail, settings.resolution) == (256, 1280)
        item["params"]["config"] = json.dumps({**defaults, "detail": 40, "resolution": "640",
            "yaw": -35, "tilt": 50, "smoothing": .8, "palette": "Inferno",
            "scene_rotation": 0, "offset_x": 0, "antialiasing": "Off"})
        image = np.repeat(np.arange(32, dtype=np.float32)[None, :, None], 24, axis=0)
        image = np.repeat(image, 3, axis=2)
        original = image.copy()
        reference = node("software", "custom", **load_folder(ROOT / "examples/custom_nodes/Thermal Surface"))
        np.testing.assert_array_equal(processor.apply(image, item), processor.apply(image, reference))
        np.testing.assert_array_equal(image, original)
        assert module._renderer.effective_execution == "OpenCV"
    finally:
        processor.close()


@pytest.mark.parametrize("failure", [ModuleNotFoundError("taichi"), RuntimeError("device unavailable")])
def test_optional_failure_returns_original_surface_once_per_selection(
    monkeypatch: Any, caplog: Any, failure: Exception,
) -> None:
    """
    Retain valid output and explain a fallback without retrying native initialization every frame.
    """
    processor, item, module = package()
    attempts = []
    image = np.full((24, 32, 3), 120, np.float32)
    try:
        expected = processor.apply(image, item)

        def unavailable(requested: str) -> None:
            """
            Simulate missing distributions and backend errors at the optional boundary.
            """
            attempts.append(requested)
            raise failure

        surface = sys.modules[module.__package__ + ".surface"]
        monkeypatch.setitem(vars(surface), "Rasterizer", unavailable)
        execution(item, "Taichi Metal")
        for _ in range(3):
            np.testing.assert_array_equal(processor.apply(image, item), expected)
        assert attempts == ["Taichi Metal"]
        assert module._renderer.effective_execution == "OpenCV"
        assert str(failure) in module._renderer.fallback_reason
        assert caplog.text.count("Thermal Surface using OpenCV") == 1
        execution(item, "OpenCV")
        processor.apply(image, item)
        execution(item, "Taichi Metal")
        processor.apply(image, item)
        assert len(attempts) == 2
    finally:
        processor.close()


def test_kernel_failure_after_success_updates_effective_backend_and_stops_retrying(monkeypatch: Any) -> None:
    """
    Handle compiler-specific exceptions after a successful frame without misreporting CPU fallback.
    """
    processor, item, module = package()
    image = np.full((24, 32, 3), 120, np.float32)
    calls = []

    class CompilerFailure(Exception):
        """
        Model an optional compiler error outside Python's ordinary runtime exception hierarchy.
        """

    class NativeRaster:
        """
        Return one valid frame, then fail exactly once at the native execution boundary.
        """

        def __init__(self, _execution: str) -> None:
            """
            Avoid native imports in deterministic tests.
            """

        @staticmethod
        def render(floor: np.ndarray, _triangles: np.ndarray,
                   _colors: np.ndarray, _wireframe: bool, _sample_scale: int = 1) -> np.ndarray:
            """
            Raise after the first frame to reproduce an already-initialized backend failure.
            """
            calls.append(True)
            if len(calls) > 1:
                raise CompilerFailure("shader compile failed")
            return floor.copy()

    try:
        expected = processor.apply(image, item)
        monkeypatch.setitem(vars(sys.modules[module.__package__ + ".surface"]), "Rasterizer", NativeRaster)
        execution(item, "Taichi CPU")
        processor.apply(image, item)
        assert module._renderer.effective_execution == "Taichi CPU"
        for _ in range(2):
            np.testing.assert_array_equal(processor.apply(image, item), expected)
        assert len(calls) == 2 and module._renderer.effective_execution == "OpenCV"
        assert "CompilerFailure" in module._renderer.fallback_reason
    finally:
        processor.close()


def test_antialiasing_averages_coverage_and_keeps_original_dimensions() -> None:
    """
    Resolve a sharp diagonal into mixed edge pixels while preserving solid interior and background.
    """
    processor, item, module = package()
    try:
        aa = sys.modules[module.__package__ + ".antialiasing"]
        floor = np.zeros((16, 16, 3), np.uint8)
        triangle = np.array([[[2.25, 2.25], [12.25, 2.25], [2.25, 12.25]]], np.float32)
        colors = np.full((1, 3), 255, np.uint8)
        original = floor.copy()
        result = aa.rasterize(floor, triangle, colors, False, 2)
        assert result.shape == floor.shape and result.dtype == np.uint8
        assert np.any((result > 0) & (result < 255))
        np.testing.assert_array_equal(result[4, 4], [255, 255, 255])
        np.testing.assert_array_equal(result[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(floor, original)
        config = {**json.loads(item["params"]["config"]), "antialiasing": "2x supersampling"}
        item["params"]["config"] = json.dumps(config)
        restored = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))["software"][2]
        image = np.full((24, 32, 3), 120, np.float32)
        smooth = processor.apply(image, restored)
        config["antialiasing"] = "Off"
        restored["params"]["config"] = json.dumps(config)
        plain = processor.apply(image, restored)
        assert smooth.shape == plain.shape and not np.array_equal(smooth, plain)
        assert np.isfinite(smooth).all()
        settings = sys.modules[module.__package__ + ".settings"]
        with pytest.raises(ValueError, match="antialiasing"):
            settings.Settings.read({"antialiasing": True})
    finally:
        processor.close()


def test_runtime_is_shared_without_resetting_other_nodes(monkeypatch: Any) -> None:
    """
    Share only a bounded kernel library while refusing incompatible backends or external resets.
    """
    processor, _, module = package()
    raster = sys.modules[module.__package__ + ".raster"]
    runtime = SimpleNamespace(prog=None)
    config = SimpleNamespace(arch="cpu")
    calls = []

    def initialize(**values: Any) -> None:
        """
        Model one native initialization without a device, downloads or process-global resets.
        """
        calls.append(values)
        runtime.prog = object()
        config.arch = values["arch"]

    ti = SimpleNamespace(cpu="cpu", metal="metal", ERROR="error", init=initialize,
                         lang=SimpleNamespace(impl=SimpleNamespace(
                             get_runtime=lambda: runtime, current_cfg=lambda: config)))
    library = object()

    def imported(name: str, _package: str | None = None) -> Any:
        """
        Resolve both real dependency boundaries without importing optional third-party code.
        """
        return ti if name == "taichi" else library

    monkeypatch.setitem(vars(raster), "importlib", SimpleNamespace(import_module=imported))
    monkeypatch.setenv("TI_SKIP_VERSION_CHECK", "ON")
    try:
        first = raster.service("Taichi CPU")
        assert raster.service("Taichi CPU") is first and first.library is library
        assert len(calls) == 1 and calls[0]["enable_fallback"] is False
        with pytest.raises(RuntimeError, match="another backend"):
            raster.service("Taichi Metal")
        assert len(calls) == 1
        digest = raster.kernel_digest
        monkeypatch.setitem(vars(raster), "kernel_digest", lambda: "updated source")
        with pytest.raises(RuntimeError, match="kernel source changed"):
            raster.service("Taichi CPU")
        monkeypatch.setitem(vars(raster), "kernel_digest", digest)
        runtime.prog = None
        with pytest.raises(RuntimeError, match="runtime changed"):
            raster.service("Taichi CPU")
        assert len(calls) == 1
    finally:
        processor.close()


def test_execution_picker_survives_refresh_locks_and_export(tmp_path: Path) -> None:
    """
    Drive the actual generated choice popup without loading or probing Taichi in the UI process.
    """
    script = '''
import json
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import validate_pipeline
app = QApplication([])
window = ViewWindow(lambda message: None)
window.show()
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
QFileDialog.getExistingDirectory = lambda *args: FOLDER
controls.open_folder()
app.processEvents()
assert json.loads(item["params"]["config"])["detail"] == 256
assert controls.fields.fields["resolution"].currentText() == "1280"
for key in ("yaw", "tilt", "scene_rotation", "offset_x", "offset_y", "zoom", "height", "smoothing"):
    assert controls.fields.fields[key].decimals() == 1
combo = controls.fields.fields["execution"]
combo.showPopup()
QTest.keyClick(combo, Qt.Key.Key_Home)
QTest.keyClick(combo, Qt.Key.Key_Down)
QTest.keyClick(combo, Qt.Key.Key_Down)
for _ in range(8):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
QTest.keyClick(combo, Qt.Key.Key_Return)
assert combo.currentText() == "Taichi Metal"
resolution = controls.fields.fields["resolution"]
resolution.showPopup()
QTest.keyClick(resolution, Qt.Key.Key_End)
for _ in range(8):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
QTest.keyClick(resolution, Qt.Key.Key_Return)
assert resolution.currentText() == "4096"
aa = controls.fields.fields["antialiasing"]
aa.showPopup()
QTest.keyClick(aa, Qt.Key.Key_End)
for _ in range(8):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
QTest.keyClick(aa, Qt.Key.Key_Return)
assert aa.currentText() == "2x supersampling"
detail = controls.fields.fields["detail"]
assert detail.maximum() == 512
detail.setFocus()
QTest.keyClick(detail, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
QTest.keyClicks(detail, "512")
QTest.keyClick(detail, Qt.Key.Key_Return)
for _ in range(8):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
saved = validate_pipeline(json.loads(json.dumps(editor.document)))
assert json.loads(saved["software"][1]["params"]["config"])["execution"] == "Taichi Metal"
assert json.loads(saved["software"][1]["params"]["config"])["detail"] == 512
assert json.loads(saved["software"][1]["params"]["config"])["resolution"] == "4096"
assert json.loads(saved["software"][1]["params"]["config"])["antialiasing"] == "2x supersampling"
editor.rebuild()
controls = editor.custom_widgets[item["id"]]
assert controls.fields.fields["execution"].currentText() == "Taichi Metal"
assert controls.fields.fields["detail"].value() == 512
assert controls.fields.fields["resolution"].currentText() == "4096"
assert controls.fields.fields["antialiasing"].currentText() == "2x supersampling"
editor.locked = True
controls.refresh()
controls.change_value("execution", "OpenCV")
controls.change_value("detail", 40)
controls.change_value("resolution", "640")
controls.change_value("antialiasing", "Off")
assert json.loads(item["params"]["config"])["execution"] == "Taichi Metal"
assert json.loads(item["params"]["config"])["detail"] == 512
assert json.loads(item["params"]["config"])["resolution"] == "4096"
assert json.loads(item["params"]["config"])["antialiasing"] == "2x supersampling"
assert not controls.fields.isEnabled()
window.close()
'''
    result = run_popup("FOLDER = " + repr(str(FOLDER)) + "\n" + script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
