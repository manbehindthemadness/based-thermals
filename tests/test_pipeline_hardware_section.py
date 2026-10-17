"""
Camera hardware disclosures change presentation without changing pipeline ownership.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_hardware_disclosures_match_environment_and_persist_independently(tmp_path: Path) -> None:
    """
    Preserve disclosure choices across refresh, rebuild, locks, camera changes and reopening.
    """
    script = r'''
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow

app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
editor = window.pipeline_editor
thermal = editor.stack_titles["hardware"]
optical = editor.stack_titles["visible_hardware"]
listing = editor.stacks["hardware"]
assert thermal.isCheckable() and not thermal.isChecked()
assert thermal.text().startswith("▸ Thermal camera hardware")
assert thermal.accessibleName() == "Thermal camera hardware"
assert thermal.styleSheet() == window.environment_button.styleSheet()
assert listing.isHidden() and optical.isHidden()
document = deepcopy(editor.document)
sent = len(messages)
QTest.mouseClick(thermal, Qt.MouseButton.LeftButton)
app.processEvents()
assert thermal.isChecked() and listing.isVisible()
assert thermal.text().startswith("▾ Thermal camera hardware")
assert listing.output_footer.isVisible()
thermal.setFocus()
QTest.keyClick(thermal, Qt.Key.Key_Space)
assert not thermal.isChecked() and listing.isHidden()
assert editor.document == document and len(messages) == sent
thermal.click()
editor.rebuild()
assert thermal.isChecked() and not listing.isHidden()
assert not window.environment_button.isChecked()
assert not window.calibrations_button.isChecked()
state = {"camera_profile": {"id": "virtual-dual", "visible": True}}
editor.update_state(state, True)
app.processEvents()
assert thermal.isEnabled() and listing.isVisible()
assert optical.isVisible() and not optical.isChecked()
assert editor.stacks["visible_hardware"].isHidden()
optical.click()
app.processEvents()
assert editor.stacks["visible_hardware"].isVisible()
thermal.click()
assert listing.isHidden() and editor.stacks["visible_hardware"].isVisible()
thermal.click()
editor.update_state(state, True)
assert editor.stacks["visible_hardware"].isVisible()
optical.click()
editor.update_state(state, True)
assert editor.stacks["visible_hardware"].isHidden()
assert listing.isVisible()
optical.click()
editor.update_state({"camera_profile": {"visible": False}}, False)
assert optical.isHidden() and editor.stacks["visible_hardware"].isHidden()
editor.update_state(state, False)
assert optical.isChecked() and not editor.stacks["visible_hardware"].isHidden()
assert len(messages) == sent and editor.document == document
window.close()
reopened = ViewWindow(messages.append)
reopened.show()
app.processEvents()
editor = reopened.pipeline_editor
assert editor.stack_titles["hardware"].isChecked()
assert editor.stacks["hardware"].isVisible()
assert not reopened.environment_button.isChecked()
assert not reopened.calibrations_button.isChecked()
assert editor.stack_titles["visible_hardware"].isHidden()
editor.update_state(state, False)
assert editor.stack_titles["visible_hardware"].isChecked()
assert not editor.stacks["visible_hardware"].isHidden()
editor.stack_titles["hardware"].click()
reopened.close()
collapsed = ViewWindow(messages.append)
assert not collapsed.pipeline_editor.stack_titles["hardware"].isChecked()
assert collapsed.pipeline_editor.stacks["hardware"].isHidden()
collapsed.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
