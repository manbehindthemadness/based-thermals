"""
Exercise every visualization palette picker in a real offscreen Qt editor subprocess.
"""

from pathlib import Path

import pytest
from support.qt_process import run_popup

FOLDERS = Path(__file__).resolve().parents[1] / "examples" / "custom_nodes"


@pytest.mark.parametrize(
    "name",
    [
        "Thermal Contours",
        "Thermal Shaded Relief",
        "Thermal History",
        "Thermal Surface",
        "Thermal Surface Taichi",
    ],
)
def test_palette_popup_during_refresh_export_rebuild_and_lock(name: str, tmp_path: Path) -> None:
    """
    Select a newly available palette with keyboard input while frame-state echoes continue.
    """
    script = """
import json
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import validate_pipeline
from based_thermals.custom_nodes.palettes import builtin_palettes
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
combo = controls.fields.fields["palette"]
assert combo.count() == len(builtin_palettes())
combo.showPopup()
QTest.keyClick(combo, Qt.Key.Key_End)
QTest.keyClick(combo, Qt.Key.Key_Up)
highlight = combo.view().currentIndex().row()
assert combo.itemData(highlight) == "camera_21"
for _ in range(12):
    editor.update_state({
        "pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial,
    }, False)
    app.processEvents()
    assert combo.view().isVisible()
    assert combo.view().currentIndex().row() == highlight
QTest.keyClick(combo, Qt.Key.Key_Return)
assert combo.currentData() == "camera_21"
values = json.loads(item["params"]["config"])
assert values["palette"] == "camera_21"
saved = validate_pipeline(json.loads(json.dumps(editor.document)))
assert json.loads(saved["software"][1]["params"]["config"]) == values
editor.rebuild()
controls = editor.custom_widgets[item["id"]]
assert controls.fields.fields["palette"].currentData() == "camera_21"
editor.locked = True
controls.refresh()
assert not controls.fields.isEnabled()
controls.change_value("palette", "Magma")
assert json.loads(item["params"]["config"]) == values
window.close()
"""
    result = run_popup("FOLDER = " + repr(str(FOLDERS / name)) + "\n" + script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
