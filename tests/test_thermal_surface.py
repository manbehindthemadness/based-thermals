"""
Verify portable thermal surface rendering without hardware or external resources.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.frames import make_frame
from support.pipeline import raw_pipeline
from support.qt_process import run_popup

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import node, validate_pipeline
from based_thermals.processing.scheduler import PipelineProcessor

FOLDER = Path(__file__).resolve().parents[1] / "examples" / "custom_nodes" / "Thermal Surface"


@pytest.fixture
def renderer() -> Any:
    """
    Load the real embedded package through the same portable runtime used by the viewer.
    """
    item = node("software", "custom", **load_folder(FOLDER))
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    try:
        yield processor, item
    finally:
        processor.close()


def configure(item: dict[str, Any], **values: Any) -> dict[str, Any]:
    """
    Update validated current configuration while preserving portable control definitions.
    """
    result = deepcopy(item)
    result["params"]["config"] = json.dumps({**json.loads(item["params"]["config"]), **values})
    return result


def test_surface_round_trip_peaks_input_ownership_and_bounded_cache(renderer: Any) -> None:
    """
    Replace pixels with a shaded mesh and refresh heights while reusing only static geometry.
    """
    processor, item = renderer
    document = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    item = document["software"][2]
    image = np.zeros((48, 64, 3), np.float32)
    image[16:28, 20:32] = 230
    original = image.copy()
    result = processor.apply(image, item)
    assert result.shape == (480, 640, 3) and result.dtype == np.float32
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
    assert np.array_equal(image, original)
    assert result.max() > 100
    package = processor.packages[item["id"]]
    module = sys.modules[package.name]
    cached = module._renderer.geometry
    wired = processor.apply(image, configure(item, wireframe=True))
    assert not np.array_equal(result, wired)
    assert module._renderer.geometry is cached
    shifted = processor.apply(np.roll(image, 16, axis=1), item)
    assert module._renderer.geometry is cached
    assert not np.array_equal(result, shifted)
    changed = processor.apply(image, configure(item, yaw=90, height=1.4, palette="Turbo"))
    assert module._renderer.geometry is not cached
    assert not np.array_equal(changed, result)
    assert len(processor.packages) == 1


def test_raw_source_integration_preserves_viewport_and_frame(renderer: Any) -> None:
    """
    Run Raw/Range/Custom/Output through the real branch scheduler and viewer rotation.
    """
    _, item = renderer
    document = validate_pipeline(raw_pipeline(item))
    frame = make_frame()
    original = bytes(frame)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, source = processor.process(frame, None, document, scale=1, rotation=90)
        assert source == "raw" and output.shape == (256, 192, 3)
        assert output.dtype == np.uint8 and np.isfinite(output).all()
        assert frame == original
    finally:
        processor.close()


@pytest.mark.parametrize("shape", [(1, 1), (2, 47), (64, 48), (48, 64)])
@pytest.mark.parametrize("normalization", ["Source range", "Auto contrast"])
def test_flat_planes_and_small_images_remain_finite(
    renderer: Any, shape: tuple[int, int], normalization: str,
) -> None:
    """
    Handle constant, portrait, narrow and single-pixel inputs without zero-span errors.
    """
    processor, item = renderer
    result = processor.apply(
        np.full((*shape, 3), 128, np.float32),
        configure(item, normalization=normalization, resolution="384", height=0, wireframe=True),
    )
    assert np.isfinite(result).all()
    assert max(result.shape[:2]) == 384
    assert result.shape[0] * result.shape[1] <= 384 * 384


def test_node_instances_and_wireframe_are_independent(renderer: Any) -> None:
    """
    Keep view/palette caches private and make mesh edges an optional visible effect.
    """
    processor, item = renderer
    image = np.full((24, 32, 3), 180, np.float32)
    first = processor.apply(image, item)
    second = configure(item, palette="Grayscale", detail=24, wireframe=True)
    second["id"] = "other-surface"
    result = processor.apply(image, second)
    assert not np.array_equal(first, result)
    packages = [sys.modules[package.name] for package in processor.packages.values()]
    assert packages[0]._renderer is not packages[1]._renderer
    assert packages[0]._renderer.geometry is not packages[1]._renderer.geometry
    assert np.array_equal(first, processor.apply(image, item))


def test_colored_preview_is_not_mistaken_for_temperature(renderer: Any) -> None:
    """
    Reject palette luminance as a thermal height proxy and explain the Raw input requirement.
    """
    processor, item = renderer
    image = np.zeros((24, 32, 3), np.float32)
    image[:, :, 2] = 200
    with pytest.raises(ValueError, match="select Source Raw"):
        processor.apply(image, item)


def test_orthographic_projection_raises_hotter_points_and_orders_depth(renderer: Any) -> None:
    """
    Check the camera against independent 45-degree projection geometry.
    """
    processor, item = renderer
    processor.apply(np.zeros((4, 4, 3), np.float32), configure(item, yaw=0, tilt=45, height=1))
    module = sys.modules[processor.packages[item["id"]].name]
    geometry = module._renderer.geometry
    projected, depth = geometry.project(np.array(((0, 0, 0), (0, 0, 1)), np.float32))
    assert projected[0, 0] == projected[1, 0]
    assert projected[0, 1] - projected[1, 1] == pytest.approx(geometry.scale / np.sqrt(2))
    assert depth[1] - depth[0] == pytest.approx(-1 / np.sqrt(2))


@pytest.mark.parametrize("rotation,right,bottom", [
    (0, True, False), (90, True, True), (180, False, True), (-90, False, False),
])
def test_asymmetric_source_keeps_correct_handedness_through_camera_roll(
    renderer: Any, rotation: float, right: bool, bottom: bool,
) -> None:
    """
    Place a hot patch at image top-right and verify its quadrant after known camera rolls.
    """
    processor, item = renderer
    image = np.zeros((48, 64, 3), np.float32)
    image[6:18, 42:56] = 255
    result = processor.apply(image, configure(
        item, yaw=0, tilt=85, height=0, smoothing=0,
        palette="Grayscale", scene_rotation=rotation,
    ))
    rows, columns = np.nonzero(result[:, :, 0] > 150)
    assert len(rows) > 0
    assert bool(columns.mean() > result.shape[1] / 2) == right
    assert bool(rows.mean() > result.shape[0] / 2) == bottom
    module = sys.modules[processor.packages[item["id"]].name]
    geometry = module._renderer.geometry
    top_count = 2 * (geometry.rows - 1) * (geometry.columns - 1)
    faces = geometry.xy[geometry.triangles[:top_count]]
    edge_a, edge_b = faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]
    upward = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    assert np.all(upward > 0)


def test_camera_roll_pan_and_zoom_transform_the_view_without_changing_depth(renderer: Any) -> None:
    """
    Check camera controls against independent screen-space transforms and fixed-depth values.
    """
    processor, item = renderer
    image = np.full((24, 32, 3), 128, np.float32)
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 0, 1)), np.float32)
    processor.apply(image, configure(item, yaw=0, tilt=45))
    module = sys.modules[processor.packages[item["id"]].name]
    base = module._renderer.geometry
    points, depth = base.project(vertices)
    processor.apply(image, configure(item, yaw=0, tilt=45, offset_x=10, offset_y=-20))
    panned, panned_depth = module._renderer.geometry.project(vertices)
    expected_pan = np.array((base.output_width * 0.1, -base.output_height * 0.2))
    np.testing.assert_allclose(panned - points, np.broadcast_to(expected_pan, points.shape))
    np.testing.assert_array_equal(panned_depth, depth)
    processor.apply(image, configure(item, yaw=0, tilt=45, zoom=2))
    zoomed, zoomed_depth = module._renderer.geometry.project(vertices)
    center = np.array((base.output_width, base.output_height)) / 2
    np.testing.assert_allclose(zoomed - center, 2 * (points - center), atol=0.0001)
    np.testing.assert_array_equal(zoomed_depth, depth)
    processor.apply(image, configure(item, yaw=0, tilt=45, scene_rotation=90))
    rolled = module._renderer.geometry
    turned, turned_depth = rolled.project(vertices)
    np.testing.assert_allclose((turned[1] - turned[0]) / rolled.scale, (0, 1), atol=1e-6)
    np.testing.assert_array_equal(turned_depth, depth)


@pytest.mark.parametrize("values", [
    {"scene_rotation": -90}, {"offset_x": 25}, {"offset_y": -25}, {"zoom": 1.5},
])
def test_camera_control_changes_replace_cached_view(renderer: Any, values: dict[str, float]) -> None:
    """
    Rebuild view/floor placement for camera changes while keeping frame dimensions stable.
    """
    processor, item = renderer
    image = np.full((24, 32, 3), 128, np.float32)
    result = processor.apply(image, item)
    module = sys.modules[processor.packages[item["id"]].name]
    original = module._renderer.geometry
    changed = processor.apply(image, configure(item, **values))
    assert changed.shape == result.shape
    assert not np.array_equal(changed, result)
    assert module._renderer.geometry is not original
    assert np.isfinite(changed).all()


def test_legacy_configuration_defaults_to_original_camera_view(renderer: Any) -> None:
    """
    Keep old configurations compatible when the package source is updated separately.
    """
    processor, item = renderer
    image = np.full((24, 32, 3), 128, np.float32)
    expected = processor.apply(image, item)
    config = json.loads(item["params"]["config"])
    for key in ("scene_rotation", "offset_x", "offset_y", "zoom"):
        del config[key]
    callback = processor.packages[item["id"]].process
    assert np.array_equal(callback(image, config), expected)


@pytest.mark.parametrize("key,value", [
    ("detail", True), ("detail", 40.5), ("detail", 200), ("yaw", None),
    ("tilt", 0), ("height", float("nan")), ("smoothing", -1),
    ("palette", []), ("normalization", "temperature"), ("resolution", "99999"),
    ("wireframe", 1),
    ("scene_rotation", 181), ("offset_x", True), ("offset_y", -101),
    ("zoom", 0), ("zoom", float("inf")),
])
def test_standalone_callback_rejects_invalid_settings(renderer: Any, key: str, value: Any) -> None:
    """
    Validate direct callback configuration before building geometry or allocating canvases.
    """
    processor, item = renderer
    processor.apply(np.zeros((2, 2, 3), np.float32), item)
    callback = processor.packages[item["id"]].process
    config = json.loads(item["params"]["config"])
    config[key] = value
    with pytest.raises((TypeError, ValueError), match="Thermal Surface:"):
        callback(np.zeros((2, 2, 3), np.float32), config)


def test_surface_editor_controls_survive_refresh_and_export(tmp_path: Path) -> None:
    """
    Exercise real pointer/keyboard editing during repeated viewer state refreshes and locks.
    """
    script = '''
import json
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import QPoint, Qt
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
fields = controls.fields.fields
assert len(fields) == 13
detail = fields["detail"]
detail.setFocus()
QTest.keyClick(detail, Qt.Key.Key_Up)
for _ in range(12):
    editor.update_state({
        "pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial,
    }, False)
    app.processEvents()
assert detail.value() == 44
for key in ("scene_rotation", "offset_x", "offset_y", "zoom"):
    fields[key].setFocus()
    QTest.keyClick(fields[key], Qt.Key.Key_Up)
    for _ in range(3):
        editor.update_state({
            "pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial,
        }, False)
        app.processEvents()
combo = fields["palette"]
combo.showPopup()
for _ in range(combo.findData("turbo") - combo.currentIndex()):
    QTest.keyClick(combo, Qt.Key.Key_Down)
for _ in range(12):
    editor.update_state({
        "pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial,
    }, False)
    app.processEvents()
QTest.keyClick(combo, Qt.Key.Key_Return)
assert combo.currentText() == "Turbo"
check = fields["wireframe"]
QTest.mouseClick(check, Qt.MouseButton.LeftButton, pos=QPoint(6, check.height() // 2))
values = json.loads(item["params"]["config"])
assert values["detail"] == 44 and values["palette"] == "turbo" and values["wireframe"]
assert values["scene_rotation"] == 5 and values["offset_x"] == values["offset_y"] == 1
assert values["zoom"] == 1.1
saved = validate_pipeline(json.loads(json.dumps(editor.document)))
editor.rebuild()
fields = editor.custom_widgets[item["id"]].fields.fields
assert fields["detail"].value() == 44
assert fields["scene_rotation"].value() == 5 and fields["zoom"].value() == 1.1
assert json.loads(saved["software"][1]["params"]["config"]) == values
editor.locked = True
controls = editor.custom_widgets[item["id"]]
controls.refresh()
assert not controls.fields.isEnabled()
controls.change_value("detail", 80)
assert json.loads(item["params"]["config"]) == values
window.close()
'''
    result = run_popup("FOLDER = " + repr(str(FOLDER)) + "\n" + script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
