"""
Portable custom gradients, atomic library imports and isolated Qt builder workflows.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.qt_process import run_popup

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.configuration import validate_controls
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.custom_nodes.visualization import palette_lut, palette_names
from based_thermals.palettes import (
    custom_lut,
    load_palettes,
    palette_token,
    read_library,
    read_palette,
    save_palettes,
    validate_palette,
    write_library,
)
from based_thermals.pipeline import default_pipeline, node, validate_pipeline
from based_thermals.pipeline_titles import node_title
from based_thermals.processing.images import colorize


@pytest.fixture
def gradient() -> dict[str, Any]:
    """
    Provide asymmetric RGB reference stops that detect channel-order errors.
    """
    return {"name": "Test gradient", "stops": [
        {"position": 0, "color": "#ff0000"}, {"position": 1, "color": "#0000ff"},
    ]}


def test_portable_gradient_rendering_and_visualization(gradient: dict[str, Any]) -> None:
    """
    Check interpolation against independent values and keep pipeline snapshots self-contained.
    """
    token = palette_token(gradient)
    lut = custom_lut(token)
    np.testing.assert_array_equal(lut[[0, 85, 170, 255]],
                                  [[0, 0, 255], [85, 0, 170], [170, 0, 85], [255, 0, 0]])
    assert not lut.flags.writeable
    detached = read_palette(token)
    detached["stops"][0]["color"] = "#ffffff"
    assert read_palette(token) == gradient
    raw = np.array([[0, 85], [170, 255]], dtype=np.uint8)
    original = raw.copy()
    np.testing.assert_array_equal(colorize(raw, token), lut[raw])
    np.testing.assert_array_equal(raw, original)
    assert palette_lut(token) is lut
    assert token in palette_names()  # Existing embedded surface validators use membership.
    controls = [{"key": "palette", "label": "Palette", "type": "choice",
                 "options": list(palette_names())}]
    assert validate_controls(controls, {"palette": token}) == controls
    with pytest.raises(ValueError):
        validate_controls([{**controls[0], "options": ["Inferno"]}], {"palette": token})
    document = default_pipeline()
    document["software"].insert(1, node("software", "colors", palette=token))
    assert validate_pipeline(json.loads(json.dumps(document))) == document
    assert node_title("software", document["software"][1]) == "App colors: Test gradient"


@pytest.mark.parametrize("name", ["Thermal Contours", "Thermal Shaded Relief", "Thermal History",
                                 "Thermal Surface", "Thermal Surface Taichi"])
def test_visualization_packages_accept_portable_palettes(
    name: str, gradient: dict[str, Any],
) -> None:
    """
    Render custom selections through real packages, including legacy surface choice checks.
    """
    folder = Path(__file__).resolve().parents[1] / "examples" / "custom_nodes" / name
    params = load_folder(folder)
    config = json.loads(params["config"])
    config.update(palette=palette_token(gradient), detail=16, samples=16, history=8,
                  resolution="384", execution="OpenCV", antialiasing="Off")
    params["config"] = json.dumps(config)
    item = node("software", "custom", **params)
    document = default_pipeline()
    document["software"].insert(1, item)
    validate_pipeline(json.loads(json.dumps(document)))
    image = np.repeat(np.linspace(0, 255, 32, dtype=np.float32)[None, :, None], 3, axis=2)
    image = np.repeat(image, 24, axis=0)
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    try:
        result = processor.apply(image, item)
        assert result.ndim == 3 and result.shape[2] == 3 and np.isfinite(result).all()
    finally:
        processor.close()


@pytest.mark.parametrize("change", [
    {"name": ""}, {"name": "a\nname"}, {"stops": None}, {"stops": []},
    {"stops": [{"position": 0, "color": "#ffffff"}]},
    {"stops": [{"position": 0, "color": "dynamic"}, {"position": 1, "color": "#ffffff"}]},
    {"stops": [{"position": 0.1, "color": "#000000"}, {"position": 1, "color": "#ffffff"}]},
    {"stops": [{"position": 0, "color": "#000000"}, {"position": 0, "color": "#ffffff"}]},
    {"stops": [{"position": False, "color": "#000000"}, {"position": 1, "color": "#ffffff"}]},
    {"stops": [{"position": 0, "color": "#000000"}, {"position": float("nan"), "color": "#ffffff"}]},
])
def test_reject_invalid_gradients(gradient: dict[str, Any], change: dict[str, Any]) -> None:
    """
    Reject malformed imports before they reach allocation or display code.
    """
    with pytest.raises(ValueError):
        validate_palette({**gradient, **change})


def test_library_round_trip_and_atomic_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gradient: dict[str, Any],
) -> None:
    """
    Keep saved data intact after invalid writes and reject malformed batch imports.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_palettes({gradient["name"]: gradient})
    assert load_palettes() == {gradient["name"]: gradient}
    path = tmp_path / "export.json"
    write_library(path, load_palettes())
    original = path.read_bytes()
    bad = deepcopy(gradient)
    bad["stops"][0]["color"] = "bad"
    with pytest.raises(ValueError):
        write_library(path, {bad["name"]: bad})
    assert path.read_bytes() == original
    for document in (None, [], {}, {"version": True, "palettes": []},
                     {"version": 1, "palettes": [gradient, gradient]},
                     {"version": 1, "palettes": [gradient, {**gradient, "name": "TEST GRADIENT"}]}):
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            read_library(path)


def test_palette_builder_and_pipeline_integration(tmp_path: Path) -> None:
    """
    Exercise save, rename, editing, import/export, locking and snapshot retention in Qt.
    """
    script = r'''
import json, sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog, QFileDialog, QMessageBox, QDoubleSpinBox
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import node
from based_thermals.palettes import read_palette, load_palettes
from based_thermals.pipeline_ui.palette_transfer import export_palette, import_palettes
from based_thermals.custom_nodes.configuration import validate_controls
from based_thermals.custom_nodes.visualization import palette_names
from based_thermals.pipeline_ui.custom_fields import CustomFields
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
builder = window.palette_builder
editor = window.pipeline_editor
assert builder.section.text().endswith('Palette builder')
assert not builder.section.isChecked()
builder.section.click()
assert builder.content.isVisible()
builder.name.setText('My palette')
builder.add_stop()
assert builder.stops.rowCount() == 3
# Embedded controls select their row, so Remove stop works after mouse interaction.
app.processEvents()
QTest.mouseClick(builder.stops.cellWidget(1, 0), Qt.MouseButton.LeftButton)
assert builder.stops.currentRow() == 1
builder.remove_stop()
assert builder.stops.rowCount() == 2
builder.add_stop()
QColorDialog.getColor = lambda *_: QColor('#00ff00')
builder.choose_color(builder.stops.cellWidget(1, 1))
builder.save()
assert builder.current == 'My palette' and builder.status.text() == 'Saved'
editor.insert('software', 'colors', 1)
item = editor.document['software'][1]
combo = editor.widgets[item['id']][1]['palette'].input
index = next(i for i in range(combo.count()) if combo.itemText(i) == 'Custom · My palette')
combo.setCurrentIndex(index)
editor.commit_pending_inputs()
old = item['params']['palette']
assert read_palette(old)['stops'][1]['color'] == '#00ff00'
# An update refreshes all active branches and marks the pipeline changed.
editor.document['branches']['B'] = [node('software', 'source'), node('software', 'colors', palette=old)]
builder.name.setText('Renamed')
builder.save()
item = editor.document['software'][1]
new = item['params']['palette']
assert old != new and read_palette(new)['name'] == 'Renamed'
assert editor.document['branches']['B'][1]['params']['palette'] == new
assert 'My palette' not in load_palettes()
# Visualization menus use opaque portable tokens while displaying friendly names.
control = {'key':'palette', 'label':'Palette', 'type':'choice', 'options':list(palette_names())}
validate_controls([control], {'palette':new})
fields = CustomFields([control], {'palette':new}, lambda *_: None)
assert fields.fields['palette'].currentData() == new
assert fields.fields['palette'].currentText() == 'Custom · Renamed'
# Export the draft and import it again after deleting the saved library entry.
destination = Path(sys.argv[1]) / 'palette.json'
QFileDialog.getSaveFileName = lambda *_: (str(destination), '')
export_palette(builder)
assert json.loads(destination.read_text())['palettes'][0]['name'] == 'Renamed'
QMessageBox.question = lambda *_: QMessageBox.StandardButton.Yes
builder.delete()
assert not load_palettes() and item['params']['palette'] == new
assert editor.widgets[item['id']][1]['palette'].input.currentData() == new
QFileDialog.getOpenFileName = lambda *_: (str(destination), '')
import_palettes(builder)
assert builder.current == 'Renamed'
# Malformed imports leave the complete library and draft intact.
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
destination.write_text('null')
import_palettes(builder)
assert warnings and 'Renamed' in load_palettes()
# Locked programmatic operations are inert, not merely disabled buttons.
before = load_palettes()
builder.set_locked(True)
builder.name.setText('Locked')
builder.save(); builder.delete(); builder.add_stop(); builder.new()
assert load_palettes() == before
# Draft switching is cancelable and does not silently lose edits.
builder.set_locked(False)
QMessageBox.question = lambda *_: QMessageBox.StandardButton.No
builder.new()
assert builder.name.text() == 'Locked'
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
