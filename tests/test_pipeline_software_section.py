"""
Software tab selection and section collapse remain distinct, display-only actions.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_software_tab_clicks_select_expand_and_collapse_without_pipeline_edits(tmp_path: Path) -> None:
    """
    Exercise actual mouse/keyboard events, refreshes and rebuilds with independent sections.
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
assert editor.stack_titles["hardware"].text() == "▸ Thermal camera hardware"
editor.insert("hardware", "brightness", 0)
editor.add_tab()
editor.add_tab()
bar = editor.tab_bar
listing = editor.stacks["software"]
editor.select_tab(1)
app.processEvents()
document = deepcopy(editor.document)
hardware_id = document["hardware"][0]["id"]
hardware_title = editor.widgets[hardware_id][3]
sent = len(messages)

def click(index: int) -> None:
    """
    Click a visible tab and process Qt selection/layout changes.
    """
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(index).center())
    app.processEvents()

def check(tab: str, expanded: bool) -> None:
    """
    Check the active branch, software body and unchanged backend document.
    """
    assert editor.tab == tab
    assert bar.tabData(bar.currentIndex()) == tab
    assert editor.software_expanded == expanded
    assert listing.isVisible() == expanded
    assert editor.tab_status.isVisible() == expanded
    assert editor.stack_titles["software"].isVisible() == expanded
    assert bar.isVisible() and window.focusWidget() is bar, (tab, expanded, window.focusWidget())
    assert editor.document == document and len(messages) == sent
    assert editor.widgets[hardware_id][3] is hardware_title

click(0)  # B open -> A open, with no extra collapse from the same click.
check("A", True)
click(1)
check("B", True)
click(1)
check("B", False)
click(1)
check("B", True)
click(1)
check("B", False)
click(0)
check("A", True)
click(0)
check("A", False)
click(2)
check("C", True)
click(2)
check("C", False)
editor.rebuild()
hardware_title = editor.widgets[hardware_id][3]
editor.update_state({}, True)
app.processEvents()
check("C", False)
click(1)  # Locks protect edits, not presentation.
check("B", True)
QTest.keyClick(bar, Qt.Key.Key_Space)
check("B", False)
QTest.keyClick(bar, Qt.Key.Key_Return)
check("B", True)
QTest.keyClick(bar, Qt.Key.Key_Left)
check("A", True)
editor.stack_titles["hardware"].click()
click(0)
check("A", False)
assert editor.stack_titles["hardware"].isChecked()
assert editor.stacks["hardware"].isVisible()
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
