"""
Portable object clipboard edits, lock guards and navigation independent of thermal measurements.
"""

from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from test_markup_editing import setup_markup

from based_thermals import desktop
from based_thermals.markup.clipboard import decode, encode, paste
from based_thermals.markup.editing import MarkupSelection
from based_thermals.markup.model import KINDS, annotation, read_annotations, serialize
from based_thermals.pipeline import node


@pytest.mark.parametrize("kind", KINDS)
def test_clipboard_preserves_styles_bindings_and_original_geometry(kind: str) -> None:
    """
    Every object kind survives JSON, receives a new identity and moves within normalized bounds.
    """
    item = annotation(kind)
    item["points"] = ([[.1, .2]] if kind == "text" else [[.1, .2], [.8, .9], [.3, .7]]
                      if kind == "polygon" else [[.1, .2], [.8, .9]])
    item.update(z_index=14, bindings={"Elapsed": "system:Session.Elapsed"})
    original = deepcopy(item)
    text = encode(item)
    assert decode(text) == item
    items = [deepcopy(item)]
    added = paste(items, text)
    assert added["id"] != item["id"] and items[0] == item == original
    assert {k: v for k, v in added.items() if k not in ("id", "points")} == {
        k: v for k, v in item.items() if k not in ("id", "points")
    }
    for before, after in zip(item["points"], added["points"], strict=True):
        assert after == pytest.approx([before[0] + .02, before[1] + .02])


@pytest.mark.parametrize("points", [[], [[0., 0.], [1., 1.]], [[.8, .8], [1., 1.]]])
def test_paste_keeps_edge_objects_in_bounds_without_resizing(points: list[list[float]]) -> None:
    """
    Full-frame and edge-aligned objects preserve size when no positive offset fits.
    """
    item = annotation("rectangle")
    item["points"] = points
    added = paste([], encode(item))
    if points:
        assert all(0 <= value <= 1 for point in added["points"] for value in point)
        assert [added["points"][1][axis] - added["points"][0][axis] for axis in (0, 1)] == pytest.approx(
            [points[1][axis] - points[0][axis] for axis in (0, 1)]
        )
    else:
        assert added["points"] == []


@pytest.mark.parametrize("payload", [None, "text", "[]", "{}", "x" * 70001,
                                     '{"format":"based-thermals-markup","version":true}',
                                     '{"format":"based-thermals-markup","version":1,"annotation":{}}'])
def test_invalid_clipboards_are_rejected_without_mutation(payload: object) -> None:
    """
    Foreign, oversized and malformed clipboard contents cannot become pipeline objects.
    """
    items = [annotation("text")]
    original = deepcopy(items)
    with pytest.raises(ValueError):
        paste(items, payload)
    assert items == original


def test_clipboard_cannot_bypass_object_or_json_limits() -> None:
    """
    A valid clipboard does not bypass the saved node's object and serialized byte budgets.
    """
    items = [annotation("text") for _ in range(128)]
    with pytest.raises(ValueError, match="128"):
        paste(items, encode(items[0]))
    assert len(items) == 128
    item = annotation("text")
    item["text"] = "a" * 8192
    items = [deepcopy(item) for _ in range(7)]
    for index, entry in enumerate(items):
        entry["id"] = str(index)
    with pytest.raises(ValueError, match="64 KiB"):
        paste(items, encode(item))
    assert len(items) == 7


@pytest.mark.usefixtures("viewer")
def test_viewer_copy_paste_delete_and_stale_menu_locks() -> None:
    """
    Route actual JSON commands, preserve dynamic text/bindings and never write hardware or spots.
    """
    session = desktop.DesktopSession(desktop, [])
    markup, item = setup_markup(session)
    item.update(text="{{ [Elapsed] :.1f }}", bindings={"Elapsed": "system:Session.Elapsed"})
    markup["params"]["annotations"] = serialize([item])
    session.markup_selection = MarkupSelection(markup["id"], item["id"])
    spots = deepcopy(session.spots.saved_state())
    revision = session.pipeline_revision
    hardware = cast(Mock, session.hardware.set)
    hardware.reset_mock()
    viewer_panel = cast(Mock, session.viewer)
    viewer_panel.poll.return_value = [{"action": "markup_object", "operation": "copy"}]
    session.poll_viewer_commands()
    response = session.markup_response
    assert decode(response["clipboard"]) == item
    viewer_panel.poll.return_value = [{"action": "markup_ack", "serial": "obsolete"}]
    session.poll_viewer_commands()
    assert session.markup_response == response
    viewer_panel.poll.return_value = [{"action": "markup_ack", "serial": response["serial"]},
                                     {"action": "markup_object", "operation": "paste", "text": response["clipboard"]}]
    session.poll_viewer_commands()
    assert not session.markup_response
    values = read_annotations(markup["params"]["annotations"])
    assert len(values) == 2 and values[1]["bindings"] == item["bindings"]
    assert session.markup_selection is not None and session.markup_selection.identity == values[1]["id"]
    assert session.selected_markup() is not None
    markup["params"]["locked"] = True
    for operation in ("copy", "paste", "delete", "edit"):
        session.markup_command({"operation": operation, "node": markup["id"],
                                "annotation": values[1]["id"], "text": response["clipboard"]})
    assert len(read_annotations(markup["params"]["annotations"])) == 2
    assert not session.markup_focus and not session.markup_response
    markup["params"]["locked"] = False
    session.markup_command({"operation": "delete"})
    assert read_annotations(markup["params"]["annotations"]) == (item,)
    assert session.markup_selection is None and session.spots.saved_state() == spots
    assert session.pipeline_revision == revision
    hardware.assert_not_called()
    viewer_panel.update.assert_not_called()  # Replies travel with normal frames, preserving queued descriptors.
    saved = desktop.load_settings()["pipeline"]
    restored = next(n for n in saved["software"] if n["id"] == markup["id"])
    assert read_annotations(restored["params"]["annotations"]) == (item,)


@pytest.mark.usefixtures("viewer")
def test_context_hit_order_double_click_and_graph_exclusion() -> None:
    """
    Context hit testing follows visible Z order, navigates branches, and excludes graph pixels.
    """
    session = desktop.DesktopSession(desktop, [])
    first, item = setup_markup(session)
    other = deepcopy(item)
    other["id"] = "upper"
    other["z_index"] = 5
    upper = node("software", "markup", annotations=serialize([other]))
    session.pipeline["branches"]["B"] = [upper]
    session.markup_frames.append({**session.markup_frames[0], "node": upper["id"], "annotation": other["id"]})
    session.show_graph = True
    session.graph_layout = desktop.GraphWindowLayout((600, 300), (400, 300))
    session.viewer_pointer({"kind": "release", "button": "right", "x": 80, "y": 40})
    assert session.markup_response["menu"] == {"node": upper["id"], "annotation": "upper"}
    assert session.markup_selection is not None and session.markup_selection.identity == "upper"
    response = deepcopy(session.markup_response)
    session.viewer_pointer({"kind": "release", "button": "right", "x": 450, "y": 40})
    assert session.markup_response == response
    imaging_panel = Mock()
    session.view_panel = imaging_panel
    session.viewer_pointer({"kind": "double", "button": "left", "x": 80, "y": 40})
    assert upper["expanded"] and session.markup_focus["node"] == upper["id"]
    assert session.markup_focus["annotation"] == "upper"
    imaging_panel.open.assert_called_once()
    state = imaging_panel.open.call_args.args[0]
    assert state["markup_focus"] == session.markup_focus
    assert session.markup_gesture.swallow_release
    upper["params"]["locked"] = first["params"]["locked"] = True
    session.viewer_pointer({"kind": "double", "button": "left", "x": 80, "y": 40})
    imaging_panel.open.assert_called_once()


@pytest.mark.usefixtures("viewer")
@pytest.mark.parametrize("lock", ["bypass", "logging_pending", "drawing"])
def test_viewer_object_commands_respect_pipeline_and_camera_locks(lock: str) -> None:
    """
    A queued menu cannot change objects after bypass, logging reservation or a new drawing session.
    """
    session = desktop.DesktopSession(desktop, [])
    markup, item = setup_markup(session)
    session.markup_selection = MarkupSelection(markup["id"], item["id"])
    original = deepcopy(session.pipeline)
    imaging_panel = Mock()
    session.view_panel = imaging_panel
    if lock == "bypass":
        markup["bypass"] = True
        original = deepcopy(session.pipeline)
    elif lock == "logging_pending":
        session.pending_save_kind = "graph_log"
    else:
        session.markup_gesture.begin(markup["id"], item, "draw")
    for operation in ("copy", "paste", "delete", "edit"):
        session.markup_command({"node": markup["id"], "annotation": item["id"],
                                "operation": operation, "text": encode(item)})
    assert session.pipeline == original
    assert not session.markup_focus and not session.markup_response
    imaging_panel.open.assert_not_called()
