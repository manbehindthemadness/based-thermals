"""
Keep Custom numeric drafts intact during repeated runtime state updates.
"""

from pathlib import Path

import pytest
from support.qt_process import run_popup

from based_thermals.custom_nodes.configuration import validate_controls


@pytest.mark.parametrize("decimals", [-1, 10, True, 1.5, "1", None])
def test_decimal_precision_rejects_invalid_metadata(decimals: object) -> None:
    """
    Reject malformed precision before controls reach Qt or saved pipeline documents.
    """
    control = {"key": "value", "label": "Value", "type": "number",
               "min": 0, "max": 10, "step": .1, "decimals": decimals}
    with pytest.raises(ValueError, match="decimal places"):
        validate_controls([control], {"value": 1})


@pytest.mark.parametrize("decimals", [0, 1, 9])
def test_decimal_precision_survives_validation(decimals: int) -> None:
    """
    Preserve explicitly requested display precision without rewriting saved numeric values.
    """
    control = {"key": "value", "label": "Value", "type": "number",
               "min": 0, "max": 10, "step": .1, "decimals": decimals}
    assert validate_controls([control], {"value": 1.2345}) == [control]


def test_keypad_numeric_drafts_survive_refresh_commit_and_locks(tmp_path: Path) -> None:
    """
    Type integer and signed decimal keypad values through the real editor while state refreshes.
    """
    script = '''
import json
from copy import deepcopy
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
item["expanded"] = True
item["params"].update(config=json.dumps({"count": 256, "offset": 0}), controls=json.dumps([
    {"key": "count", "label": "Count", "type": "integer", "min": 12, "max": 512, "step": 4},
    {"key": "offset", "label": "Offset", "type": "number", "min": -100, "max": 100,
     "step": .1, "decimals": 1},
]))
editor.rebuild()
window.show()
app.processEvents()
controls = editor.custom_widgets[item["id"]]
assert controls.fields.fields["offset"].decimals() == 1
def refresh():
    for _ in range(3):
        editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
        app.processEvents()
def press(field, key, text):
    for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        app.sendEvent(field, QKeyEvent(event_type, key, Qt.KeyboardModifier.KeypadModifier, text))
for name, keys, value in (
    ("count", [(Qt.Key.Key_1, "1"), (Qt.Key.Key_2, "2"), (Qt.Key.Key_8, "8")], 128),
    ("offset", [(Qt.Key.Key_Minus, "-"), (Qt.Key.Key_1, "1"), (Qt.Key.Key_4, "4"),
                (Qt.Key.Key_Period, "."), (Qt.Key.Key_5, "5")], -14.5),
):
    field = controls.fields.fields[name]
    field.setFocus()
    app.processEvents()
    field.selectAll()
    sent = len(messages)
    draft = ""
    for key, text in keys:
        press(field, key, text)
        draft += text
        refresh()
        assert field.cleanText() == draft, (name, draft, field.cleanText())
        assert len(messages) == sent
    press(field, Qt.Key.Key_Enter, "\\r")
    refresh()
    assert json.loads(item["params"]["config"])[name] == value
    if name == "offset":
        assert field.cleanText() == "-14.5"
# Save-time commit includes a draft even before Enter or focus loss.
field = controls.fields.fields["offset"]
field.setFocus()
field.selectAll()
press(field, Qt.Key.Key_2, "2")
refresh()
editor.commit_pending_inputs()
assert json.loads(item["params"]["config"])["offset"] == 2
# Explicit replacement and locks supersede a focused draft.
field.selectAll()
press(field, Qt.Key.Key_5, "5")
controls.set_config('{"count":128,"offset":3}')
assert field.value() == 3
field.selectAll()
press(field, Qt.Key.Key_6, "6")
editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, True)
assert field.value() == 3 and not field.isEnabled()
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
