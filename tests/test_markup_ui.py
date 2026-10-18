"""
Native Markup controls and main-process drawing through the existing JSON boundaries.
"""

from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.frames import make_frame
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.markup.labels import IDENTITY
from based_thermals.markup.model import annotation, read_annotations, serialize
from based_thermals.pipeline import node


def test_markup_controls_refresh_focus_popup_locks_and_portability(tmp_path: Path) -> None:
    """
    Exercise actual Qt controls during repeated incoming updates and restore saved annotations.
    """
    script = '''
import json
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidget
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import validate_pipeline
from based_thermals.markup.model import read_annotations
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
editor.insert("software", "markup", 1)
item = editor.document["software"][1]
controls = editor.markup_widgets[item["id"]]
assert isinstance(controls.objects, QListWidget)
controls.add.click()
app.processEvents()
assert messages[-1]["action"] == "markup_draw"
controls.text.setFocus()
controls.text.selectAll()
QTest.keyClicks(controls.text, "Input stage")
for _ in range(8):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
assert controls.text.toPlainText() == "Input stage"
controls.numbers["font_size"].setFocus()
QTest.keyClick(controls.numbers["font_size"], Qt.Key.Key_Up)
combo = controls.colors["text"].combo
combo.showPopup()
for _ in range(8):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
    app.processEvents()
assert combo.view().isVisible()
combo.hidePopup()
controls.add.click()
controls.z_index.setValue(8)
controls.objects.setCurrentRow(0)
controls.bring_front.click()
assert controls.z_index.value() == 9
controls.send_back.click()
assert controls.z_index.value() == 7
controls.z_index.setFocus()
controls.z_index.lineEdit().selectAll()
QTest.keyClicks(controls.z_index, "-12")
controls.bring_front.click()
assert controls.z_index.value() == 9
controls.objects.setCurrentRow(1)
controls.z_index.setValue(1000)
controls.objects.setCurrentRow(0)
controls.bring_front.click()
assert controls.z_index.value() == 9 and "limit reached" in controls.layer_status.text()
controls.lock.setChecked(True)
assert not controls.draw.isEnabled() and not controls.add.isEnabled()
assert controls.fields.isEnabled()
saved = validate_pipeline(json.loads(json.dumps(editor.document)))
values = read_annotations(saved["software"][1]["params"]["annotations"])
assert values[0]["text"] == "Input stage" and values[0]["font_size"] == 17
assert values[0]["z_index"] == 9 and values[1]["z_index"] == 1000
assert saved["software"][1]["params"]["locked"]
editor.rebuild()
controls = editor.markup_widgets[item["id"]]
assert controls.text.toPlainText() == "Input stage" and controls.lock.isChecked()
# Viewer selection updates the visible list without emitting another selection request.
count = len(messages)
second_id = values[1]["id"]
editor.update_state({"pipeline": deepcopy(saved), "pipeline_serial": editor.edit_serial,
                     "markup_selection": {"node": item["id"], "annotation": second_id}}, False)
assert controls.selected_id() == second_id and len(messages) == count
controls.objects.setCurrentRow(0)
assert messages[-1] == {"action": "markup_select", "node": item["id"], "annotation": values[0]["id"]}
editor.update_state({"pipeline": deepcopy(saved), "pipeline_serial": editor.edit_serial,
                     "markup_selection": {"node": item["id"], "annotation": values[0]["id"]}}, False)
assert controls.selected_id() == values[0]["id"]
editor.update_state({"pipeline": deepcopy(saved), "pipeline_serial": editor.edit_serial}, True)
assert not controls.isEnabled()
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.usefixtures("viewer")
@pytest.mark.parametrize("graph", [False, True])
def test_draw_drag_lock_and_persist_without_measurement_or_hardware_changes(graph: bool) -> None:
    """
    Draw and drag through real viewer events even on an unregistered complex pipeline image.
    """
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock()
    session.spots.placing = False
    session.show_graph = graph
    session.graph_layout = desktop.GraphWindowLayout((600, 300), (400, 300)) if graph else None
    session.rendered = session.api.replace(session.renderer.render_detailed(make_frame()),
                                           measurement_mapping_valid=False)
    item = annotation("rectangle")
    markup = node("software", "markup", annotations=serialize([item]))
    session.set_pipeline({**session.pipeline, "software": [*session.pipeline["software"][:-1],
                                                         markup, session.pipeline["software"][-1]]})
    markup = session.markup_node(markup["id"])
    assert markup is not None
    session.markup_frames = [{"node": markup["id"], "annotation": "", "matrix": list(IDENTITY)}]
    original_spots = deepcopy(session.spots.saved_state())
    revision = session.pipeline_revision
    hardware_set = cast(Mock, session.hardware.set)
    hardware_set.reset_mock()
    session.begin_markup({"node": markup["id"], "annotation": item["id"]})

    def pointer(kind: str, x: int, y: int) -> None:
        """
        Deliver normalized-to-bitmap pointer commands through the viewer validator.
        """
        session.viewer_pointer({"kind": kind, "button": "left", "x": x, "y": y, "flags": 0})

    pointer("press", 100, 100)
    pointer("move", 200, 200)
    assert not read_annotations(markup["params"]["annotations"])[0]["points"]
    source = np.zeros_like(session.rendered.image)
    assert np.any(session.draw_markup(source)) and not np.any(source)
    pointer("release", 200, 200)
    session.markup_action("draw_finish")
    first = deepcopy(read_annotations(markup["params"]["annotations"])[0]["points"])
    assert len(first) == 2
    session.markup_frames.append({"node": markup["id"], "annotation": item["id"], "matrix": list(IDENTITY)})
    pointer("press", 150, 150)
    pointer("move", 180, 170)
    pointer("release", 180, 170)
    moved = deepcopy(read_annotations(markup["params"]["annotations"])[0]["points"])
    assert moved != first and not session.markup_gesture.swallow_release
    markup["params"]["locked"] = True
    pointer("press", 180, 170)
    pointer("move", 200, 190)
    pointer("release", 200, 190)
    assert read_annotations(markup["params"]["annotations"])[0]["points"] == moved
    with pytest.raises(ValueError, match="Unlock"):
        session.begin_markup({"node": markup["id"], "annotation": item["id"]})
    assert session.spots.saved_state() == original_spots
    assert session.pipeline_revision == revision
    hardware_set.assert_not_called()
    persisted = desktop.load_settings()
    restored = next(n for n in persisted["pipeline"]["software"] if n["type"] == "markup")
    assert read_annotations(restored["params"]["annotations"])[0]["points"] == moved


@pytest.mark.usefixtures("viewer")
def test_live_z_order_updates_paint_selection_and_drag_preview_without_worker_refresh() -> None:
    """
    Resolve current Z values rather than stale worker ordering, including active drag previews.
    """
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock()
    session.show_graph = False
    session.spots.placing = False
    session.rendered = session.renderer.render_detailed(make_frame())
    items = []
    for color, z_index in (("#ff0000", 10), ("#0000ff", 0)):
        item = annotation("rectangle")
        item.update(points=[[.1, .1], [.5, .5]], fill=True, fill_color=color,
                    fill_opacity=1., z_index=z_index)
        items.append(item)
    markup = node("software", "markup", annotations=serialize(items))
    session.pipeline["software"].insert(1, markup)
    session.markup_frames = [
        {"node": markup["id"], "annotation": item["id"], "matrix": list(IDENTITY)} for item in items
    ]
    source = np.zeros_like(session.rendered.image)
    assert session.draw_markup(source)[150, 150].tolist() == [0, 0, 255]
    session.viewer_pointer({"kind": "press", "button": "left", "x": 150, "y": 150})
    assert session.markup_gesture.identity == items[0]["id"]
    items[0]["z_index"] = -10
    markup["params"]["annotations"] = serialize(items)
    assert session.draw_markup(source)[150, 150].tolist() == [255, 0, 0]
    session.markup_gesture.cancel()
    session.markup_gesture.swallow_release = False
    session.viewer_pointer({"kind": "press", "button": "left", "x": 150, "y": 150})
    assert session.markup_gesture.identity == items[1]["id"]
