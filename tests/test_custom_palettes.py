"""
App palette access for arbitrary Custom package controls and worker callbacks.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from support.qt_process import run_popup

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.configuration import validate_controls
from based_thermals.custom_nodes.palettes import (
    apply_palette,
    builtin_palettes,
    palette_choices,
    palette_key,
    palette_lut,
)
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.palettes import palette_token, save_palettes
from based_thermals.pipeline import default_pipeline, node, validate_pipeline
from based_thermals.processing.images import colorize


def custom_gradient() -> dict:
    """
    Supply independent red-to-blue endpoints for portable worker selections.
    """
    return {"name": "Test ramp", "stops": [
        {"position": 0, "color": "#ff0000"}, {"position": 1, "color": "#0000ff"},
    ]}


def test_app_palette_api_discovery_and_display_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Share exact built-in outputs and discover saved gradients without mutating input planes.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    gradient = custom_gradient()
    save_palettes({gradient["name"]: gradient})
    token = palette_token(gradient)
    assert (token, "Custom · Test ramp") in palette_choices()
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    for key, _label in builtin_palettes():
        np.testing.assert_array_equal(apply_palette(ramp, key), colorize(ramp, key))
        assert not palette_lut(key).flags.writeable
    assert palette_key("Turbo") == "turbo"
    assert palette_key("Grayscale") == "white_hot"
    assert palette_lut("Turbo") is palette_lut("turbo")
    raw = np.array([[-10, 85, 170, 300]], dtype=np.float32)
    before = raw.copy()
    result = apply_palette(raw, token)
    np.testing.assert_array_equal(result, [[[0, 0, 255], [85, 0, 170], [170, 0, 85], [255, 0, 0]]])
    np.testing.assert_array_equal(raw, before)
    save_palettes({})
    np.testing.assert_array_equal(apply_palette(raw, token), result)


@pytest.mark.parametrize("value", [None, 1, "missing", "custom:null"])
def test_palette_control_validation(value: object) -> None:
    """
    Reject malformed palette selections without treating them as arbitrary strings.
    """
    with pytest.raises(ValueError):
        validate_controls([{"key": "ramp", "label": "Colors", "type": "palette"}], {"ramp": value})


@pytest.mark.parametrize("image", [np.array([]), np.ones((1, 1, 3)),
                                   np.array([[float("nan")]]), np.array([[1j]])])
def test_palette_api_rejects_invalid_planes(image: np.ndarray) -> None:
    """
    Validate shape, numeric type and finite display values before LUT indexing.
    """
    with pytest.raises(ValueError):
        apply_palette(image, "turbo")


def test_palette_package_load_save_and_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Execute a package with no embedded catalog and an arbitrary palette-control key.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    folder = tmp_path / "Shared colors"
    folder.mkdir()
    (folder / "__init__.py").write_text('''
from based_thermals.custom_nodes.palettes import apply_palette
CONFIG_JSON = {
    "defaults": {"ramp": "turbo"},
    "controls": [{"key": "ramp", "label": "Colors", "type": "palette"}],
}
def process(image, config):
    return apply_palette(image[:, :, 0], config["ramp"])
''')
    params = load_folder(folder)
    assert json.loads(params["controls"]) == [{"key": "ramp", "label": "Colors", "type": "palette"}]
    item = node("software", "custom", **params)
    image = np.repeat(np.array([[[0], [255]]], dtype=np.float32), 3, axis=2)
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    try:
        np.testing.assert_array_equal(processor.apply(image, item), apply_palette(image[:, :, 0], "turbo"))
        token = palette_token(custom_gradient())
        item["params"]["config"] = json.dumps({"ramp": token})
        document = default_pipeline()
        document["software"].insert(1, item)
        loaded = validate_pipeline(json.loads(json.dumps(document)))["software"][1]
        np.testing.assert_array_equal(processor.apply(image, loaded), [[[0, 0, 255], [255, 0, 0]]])
        assert len(processor.packages) == 1
    finally:
        processor.close()


def test_palette_picker_popup_refresh_and_editor_lock(tmp_path: Path) -> None:
    """
    Preserve popup highlights through frame updates, discover new entries and honor locks.
    """
    script = r'''
from PySide6.QtWidgets import QApplication
from based_thermals.pipeline_ui.custom_fields import CustomFields
from based_thermals.custom_nodes.configuration import validate_controls
from based_thermals.palettes import save_palettes, palette_token
app = QApplication([])
control = {'key':'ramp', 'label':'Colors', 'type':'palette'}
values = {'ramp':'inferno'}
validate_controls([control], values)
changes = []
fields = CustomFields([control], values, lambda *args: changes.append(args))
fields.show()
app.processEvents()
picker = fields.fields['ramp']
picker.showPopup()
app.processEvents()
assert picker.view().isVisible()
index = picker.findData('turbo')
picker.view().setCurrentIndex(picker.model().index(index, 0))
count = picker.count()
new = {'name':'New ramp', 'stops':[{'position':0, 'color':'#000000'}, {'position':1, 'color':'#ffffff'}]}
save_palettes({'New ramp':new})
for _ in range(5):
    fields.refresh(values)
    app.processEvents()
assert not changes and picker.count() == count
assert picker.view().currentIndex().row() == index
picker.hidePopup()
picker.setCurrentIndex(index)
picker.activated.emit(index)
assert changes == [('ramp','turbo')]
values['ramp'] = 'turbo'
picker.showPopup()
app.processEvents()
token = palette_token(new)
assert picker.findData(token) >= 0
picker.hidePopup()
fields.setEnabled(False)
picker.setCurrentIndex(picker.findData(token))
picker.activated.emit(picker.currentIndex())
assert changes == [('ramp','turbo')]
# Explicit state replacement remains valid even if the local library entry is deleted.
save_palettes({})
fields.refresh({'ramp':token}, preserve_pending=False)
assert picker.currentData() == token
fields.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


def test_builder_updates_arbitrary_palette_control_keys(tmp_path: Path) -> None:
    """
    Update declared palette fields across active tabs while preserving unrelated config values.
    """
    script = r'''
import json
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import node
from based_thermals.palettes import palette_token
from based_thermals.pipeline_ui.palette_transfer import refresh_pipeline_palettes
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
old = palette_token({'name':'Old', 'stops':[{'position':0, 'color':'#000000'}, {'position':1, 'color':'#ffffff'}]})
new = palette_token({'name':'New', 'stops':[{'position':0, 'color':'#ff0000'}, {'position':1, 'color':'#0000ff'}]})
item = node('software', 'custom', config=json.dumps({'colors':old, 'unrelated':old}),
            controls=json.dumps([{'key':'colors', 'label':'Colors', 'type':'palette'}]))
editor.document['branches']['B'] = [node('software', 'source'), item]
refresh_pipeline_palettes(editor, old, new)
config = json.loads(item['params']['config'])
assert config == {'colors':new, 'unrelated':old}
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
