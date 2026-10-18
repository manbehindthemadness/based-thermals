"""
Real Qt object menus/shortcuts and one-shot Imaging navigation in isolated processes.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_native_object_clipboard_menus_shortcuts_and_text_scope(tmp_path: Path) -> None:
    """
    Exercise platform shortcuts and menu actions, lock changes and ordinary text-field paste.
    """
    script = '''
from copy import deepcopy
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
from based_thermals.viewer_window import ViewerWindow
from based_thermals.markup.clipboard import decode, encode
from based_thermals.markup.model import read_annotations
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
editor.insert("software", "markup", 1)
node = editor.document["software"][1]
controls = editor.markup_widgets[node["id"]]
controls.add.click()
controls.text.setPlainText("Copied annotation")
controls.objects.setFocus()
app.processEvents()
QTest.keySequence(controls.objects, QKeySequence(QKeySequence.StandardKey.Copy))
original = decode(app.clipboard().text())
QTest.keySequence(controls.objects, QKeySequence(QKeySequence.StandardKey.Paste))
values = controls.items()
assert len(values) == 2 and values[0] == original and values[1]["id"] != original["id"]
assert controls.selected_id() == values[1]["id"]
controls.object_menu(controls.objects.visualItemRect(controls.objects.item(1)).center())
menu = controls.object_actions.menu
assert menu.isVisible()
# Incoming frame state must not close or replace the menu.
editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
assert menu.isVisible()
next(a for a in menu.actions() if a.text() == "Delete").trigger()
menu.close()
assert len(controls.items()) == 1
# A menu opened before locking cannot mutate even when its old QAction was enabled.
controls.object_menu(QPoint(5, 5))
menu = controls.object_actions.menu
controls.lock.setChecked(True)
next(a for a in menu.actions() if a.text() == "Paste").trigger()
menu.close()
assert len(controls.items()) == 1 and not controls.remove.isEnabled()
controls.objects.setFocus()
QTest.keyClick(controls.objects, Qt.Key.Key_Delete)
QTest.keyClick(controls.objects, Qt.Key.Key_Backspace)
assert len(controls.items()) == 1
controls.lock.setChecked(False)
# Normal text paste never inserts/deletes a pipeline object.
controls.text.setFocus()
app.clipboard().setText("ordinary text")
controls.text.selectAll()
QTest.keySequence(controls.text, QKeySequence(QKeySequence.StandardKey.Paste))
assert controls.text.toPlainText() == "ordinary text" and len(controls.items()) == 1
QTest.keyClick(controls.text, Qt.Key.Key_Backspace)
assert len(controls.items()) == 1
controls.objects.setFocus()
QTest.keyClick(controls.objects, Qt.Key.Key_Backspace)
assert not controls.items()

viewer_messages = []
viewer = ViewerWindow(viewer_messages.append)
viewer.show()
state = {"unit": "C", "markup_editable": True, "markup_selected": True}
viewer.update_state(state)
viewer.canvas.image = QImage(200, 100, QImage.Format.Format_BGR888)
viewer.canvas.setFocus()
app.processEvents()
QTest.keySequence(viewer.canvas, QKeySequence(QKeySequence.StandardKey.Copy))
assert viewer_messages[-1]["operation"] == "copy"
reply = {"serial": "copy1", "clipboard": encode(original)}
viewer.update_state({**state, "markup_response": reply})
assert decode(app.clipboard().text()) == original
app.clipboard().setText("new clipboard")
viewer.update_state({**state, "markup_response": reply})
assert app.clipboard().text() == "new clipboard"  # repeated state must not overwrite newer copies
app.clipboard().setText(encode(original))
QTest.keySequence(viewer.canvas, QKeySequence(QKeySequence.StandardKey.Paste))
assert viewer_messages[-1]["operation"] == "paste" and decode(viewer_messages[-1]["text"]) == original
viewer.update_state({**state, "markup_response": {"serial": "menu1", "menu": {"node": "n", "annotation": "a"}}})
menu = viewer.markup_actions.menu
assert menu.isVisible()
next(a for a in menu.actions() if a.text() == "Delete").trigger()
assert viewer_messages[-1]["operation"] == "delete" and viewer_messages[-1]["node"] == "n"
menu.close()
app.processEvents()
assert not viewer._markup_target
viewer.canvas.setFocus()
QTest.mouseDClick(viewer.canvas, Qt.MouseButton.LeftButton, pos=viewer.canvas.rect().center())
assert any(m.get("kind") == "double" for m in viewer_messages)
viewer.update_state({**state, "markup_editable": False})
before = len(viewer_messages)
QTest.keyClick(viewer.canvas, Qt.Key.Key_Delete)
assert not any(m.get("action") == "markup_object" for m in viewer_messages[before:])
# Backspace retains drawing undo instead of being captured as object deletion.
viewer.update_state({**state, "measurement_drawing": {"active": True, "kind": "polygon", "hint": "draw"}})
before = len(viewer_messages)
QTest.keySequence(viewer.canvas, QKeySequence(QKeySequence.StandardKey.Copy))
assert len(viewer_messages) == before
QTest.keyClick(viewer.canvas, Qt.Key.Key_Backspace)
assert viewer_messages[-1]["action"] == "draw_undo"
window.close()
viewer.setProperty("parent_disconnected", True)
viewer.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_double_click_focus_expands_correct_branch_node_and_element(tmp_path: Path) -> None:
    """
    Navigate once by stable IDs even from a collapsed different tab, without stealing later focus.
    """
    script = '''
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import node
from based_thermals.markup.model import annotation, serialize
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
first, second = annotation("text"), annotation("textbox")
first["text"], second["text"] = "first", "edit this"
markup = node("software", "markup", annotations=serialize([first, second]))
markup["expanded"] = True  # The desktop opens the owning node before publishing the focus request.
doc = deepcopy(editor.document)
doc["branches"]["B"] = [node("software", "source"), markup]
editor.set_software_expanded(False)
request = {"serial": "focus1", "node": markup["id"], "annotation": second["id"]}
state = {"pipeline": doc, "pipeline_serial": editor.edit_serial, "markup_focus": request,
         "markup_selection": {"node": markup["id"], "annotation": second["id"]}}
window.update_state(state)
QTest.qWait(20)
assert editor.tab == "B" and editor.software_expanded
controls = editor.markup_widgets[markup["id"]]
assert controls.isVisible() and controls.selected_id() == second["id"]
assert controls.text.hasFocus() and controls.text.toPlainText() == "edit this"
assert not any(m.get("action") == "markup_select" for m in messages)
controls.objects.setFocus()
window.update_state(state)
app.processEvents()
assert editor.markup_widgets[markup["id"]] is controls and controls.objects.hasFocus()
controls.objects.setCurrentRow(0)
request = {**request, "serial": "focus2"}
window.update_state({**state, "markup_focus": request})
app.processEvents()
controls = editor.markup_widgets[markup["id"]]
assert controls.selected_id() == second["id"] and controls.text.hasFocus()
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
