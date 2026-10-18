"""
Actual offscreen variable lookup, formula/table dialogs, full-width editing and viewer telemetry.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_variable_formula_table_editing_export_and_focus(tmp_path: Path) -> None:
    """
    Drive real modal menus and fields, including saved bindings and ordinary state echoes.
    """
    script = '''
from copy import deepcopy
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QContextMenuEvent, QTextCursor
from PySide6.QtWidgets import QApplication, QTableWidgetItem
from based_thermals.view_window import ViewWindow
from based_thermals.markup.model import read_annotations
from based_thermals.markup.expressions import render_item
from based_thermals.pipeline_ui.markup_text import VariablePicker, FormulaDialog, TableDialog
app = QApplication([])
window = ViewWindow(lambda message: None)
window.resize(1100, 1100)
window.show()
editor = window.pipeline_editor
editor.insert("software", "markup", 1)
item = editor.document["software"][1]
controls = editor.markup_widgets[item["id"]]
controls.add.click()
state = {"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial,
         "markup_live": {"Temperature.Unit": "C", "Temperature.Maximum": 35., "Temperature.Minimum": 20.},
         "markup_measurements": [{"number": 1, "name": "Desk"}]}
editor.update_state(state, False)
controls.text.setPlainText("Peak ")
cursor = controls.text.textCursor()
cursor.movePosition(QTextCursor.MoveOperation.End)
controls.text.setTextCursor(cursor)
failures = []
def drive(callback):
    """Close test dialogs even when a callback assertion fails, retaining the original failure."""
    def run():
        """Deliver deterministic input on the modal event loop without camera or user dialogs."""
        dialog = app.activeModalWidget()
        try:
            callback(dialog)
        except Exception as exc:
            failures.append(repr(exc))
            if dialog is not None:
                dialog.reject()
    QTimer.singleShot(0, run)
def choose_maximum(dialog):
    """Search and choose a real current-value leaf, never a group heading."""
    assert isinstance(dialog, VariablePicker)
    dialog.search.setText("maximum")
    leaves = [group.child(i) for group in dialog.groups.values() for i in range(group.childCount())]
    leaf = next(leaf for leaf in leaves if leaf.data(0, 256)["key"] == "Temperature.Maximum")
    assert not leaf.isHidden() and leaf.text(1) == "35"
    dialog.tree.setCurrentItem(leaf)
    dialog.accept()
def choose_text_action():
    """Choose Add variable on the actual text context menu before opening its modal dialog."""
    popup = app.activePopupWidget()
    action = next(action for action in popup.actions() if action.data() == "variable")
    popup.setActiveAction(action)
    QTest.keyClick(popup, Qt.Key.Key_Return)
    drive(choose_maximum)
controls.text.setFocus()
QTimer.singleShot(0, choose_text_action)
position = QPoint(4, 4)
controls.text.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, position,
                                               controls.text.mapToGlobal(position)))
assert not failures, failures
assert controls.text.toPlainText() == "Peak {{ [Temperature.Maximum] }}"
def formula(dialog):
    """Use the formula editor's normal field and format selector with immediate numeric preview."""
    assert isinstance(dialog, FormulaDialog)
    dialog.expression.setPlainText("[Temperature.Maximum] - [Temperature.Minimum]")
    dialog.formatting.setCurrentIndex(dialog.formatting.findData(".1f"))
    assert dialog.preview.text() == "Preview: 15.0"
    dialog.accept()
drive(formula)
controls.text.add_formula()
assert not failures, failures
saved = read_annotations(item["params"]["annotations"])[0]
assert saved["bindings"]["Temperature.Maximum"] == "Temperature.Maximum"
assert "15.0" in render_item(saved, state["markup_live"])["text"]
def table(dialog):
    """Edit actual cells, including dynamic data and formulas, before atomically accepting."""
    assert isinstance(dialog, TableDialog)
    dialog.rows.setValue(3)
    dialog.columns.setValue(2)
    rows = [["Name", "Value"], ["Max", "{{ [Temperature.Maximum] :.1f }}"],
            ["Delta", "{{ [Temperature.Maximum] - [Temperature.Minimum] :.1f }}"]]
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            dialog.table.setItem(row_index, column, QTableWidgetItem(value))
    app.processEvents()
    dialog.table.setItem(2, 1, QTableWidgetItem(""))
    def choose_cell_action():
        """Exercise the actual right-click cell menu and its nested variable lookup."""
        popup = app.activePopupWidget()
        try:
            action = next(action for action in popup.actions() if action.text().startswith("Add variable"))
            popup.setActiveAction(action)
            QTest.keyClick(popup, Qt.Key.Key_Return)
            drive(choose_maximum)
        except Exception as exc:
            failures.append(repr(exc))
        finally:
            if popup is not None:
                popup.close()
    QTimer.singleShot(0, choose_cell_action)
    dialog.context_menu(dialog.table.visualItemRect(dialog.table.item(2, 1)).center())
    assert dialog.table.item(2, 1).text() == "{{ [Temperature.Maximum] }}"
    dialog.table.setItem(2, 1, QTableWidgetItem(rows[2][1]))
    dialog.accept()
drive(table)
controls.edit_table()
assert not failures, failures
saved = read_annotations(item["params"]["annotations"])[0]
assert saved["kind"] == "table" and saved["text"].count("\\t") == 3
assert render_item(saved, state["markup_live"])["text"] == "Name\\tValue\\nMax\\t35.0\\nDelta\\t15.0"
app.processEvents()
assert abs(controls.text.width() - controls.objects.width()) <= 4
controls.text.setFocus()
caret = controls.text.textCursor().position()
for _ in range(8):
    editor.update_state({**state, "pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
assert controls.text.textCursor().position() == caret
editor.rebuild()
restored = editor.markup_widgets[item["id"]]
assert restored.text.toPlainText() == saved["text"] and restored.variable_bindings() == saved["bindings"]
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_viewer_paint_metrics_are_sampled_once_and_throttled(tmp_path: Path) -> None:
    """
    Publish genuine paint counters and frame-linked timing averages at most twice per second.
    """
    script = '''
from PySide6.QtWidgets import QApplication
from based_thermals import viewer_window
app = QApplication([])
clock = [0.]
viewer_window.monotonic = lambda: clock[0]
messages = []
window = viewer_window.ViewerWindow(messages.append)
window._viewer_state = {"readout_metrics": True}
for index in range(20):
    clock[0] = (index + 1) * .04
    window._frame_paint_pending = True
    window._frame_draw_ms = 2.
    window._frame_copy_ms = 1.
    window._captured_at = clock[0] - .025
    window._viewport_painted(3.)
    window._viewport_painted(10.)
assert len(messages) == 1
message = messages[0]
assert message["action"] == "viewer_metrics" and message["frames"] == 13
assert abs(message["fps"] - 25.) < .01
assert message["draw_ms"] == 6. and abs(message["total_ms"] - 25.) < .001
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
