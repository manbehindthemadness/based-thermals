"""
Annotation control dragging, persistent selection and oriented keyboard nudges without sampling.
"""

from copy import deepcopy
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.frames import make_frame

from based_thermals import desktop
from based_thermals.markup.editing import MarkupSelection, handles, move_handle
from based_thermals.markup.interaction import MarkupGesture
from based_thermals.markup.labels import IDENTITY
from based_thermals.markup.model import annotation, read_annotations, serialize
from based_thermals.markup.paint import project
from based_thermals.pipeline import node


@pytest.mark.parametrize("kind", ["rectangle", "textbox", "ellipse"])
def test_box_handles_keep_opposite_corner_and_clip_to_canvas(kind: str) -> None:
    """
    All four bounds corners resize without translating the opposite corner.
    """
    item = annotation(kind)
    item["points"] = [[.2, .3], [.7, .8]]
    assert handles(item) == [[.2, .3], [.7, .3], [.7, .8], [.2, .8]]
    assert move_handle(item, 1, [.9, -.1]) == [[.9, 0.], [.2, .8]]
    assert item["points"] == [[.2, .3], [.7, .8]]


@pytest.mark.parametrize("kind,points,index", [
    ("polygon", [[.1, .1], [.6, .2], [.4, .8]], 1),
    ("arrow", [[.1, .1], [.6, .2]], 1),
    ("line", [[.1, .1], [.6, .2]], 0),
    ("text", [[.1, .1]], 0),
])
def test_individual_controls_preserve_other_points_and_drag_offset(
    kind: str, points: list[list[float]], index: int,
) -> None:
    """
    A slightly off-center handle press must not snap the point to the mouse position.
    """
    item = annotation(kind)
    item["points"] = deepcopy(points)
    gesture = MarkupGesture()
    gesture.begin("node", item, "drag", IDENTITY)
    gesture.handle = index
    gesture.pointer("press", [.3, .3])
    gesture.pointer("move", [.35, .4])
    expected = deepcopy(points)
    expected[index] = (np.asarray(points[index]) + [.05, .1]).tolist()
    np.testing.assert_allclose(gesture.geometry(), expected)
    assert item["points"] == points


def setup_markup(session: Any, matrix: tuple[float, ...] = IDENTITY) -> tuple[dict, dict]:
    """
    Publish a fixed rectangle through a fake completed worker result and a real desktop session.
    """
    session.viewer = Mock()
    session.viewer.poll.return_value = []
    session.show_graph = False
    session.spots.placing = False
    session.rendered = session.api.replace(session.renderer.render_detailed(make_frame()),
                                           measurement_mapping_valid=False)
    item = annotation("rectangle")
    item["points"] = [[.1, .1], [.5, .5]]
    markup = node("software", "markup", annotations=serialize([item]))
    session.pipeline["software"].insert(1, markup)
    session.markup_frames = [{"node": markup["id"], "annotation": item["id"], "matrix": list(matrix)}]
    return markup, item


@pytest.mark.usefixtures("viewer")
def test_select_drag_vertex_nudge_lock_and_handle_pixel_ownership() -> None:
    """
    Persist selected control edits on release, route actual arrow messages, and honor locks.
    """
    session = desktop.DesktopSession(desktop, [])
    markup, item = setup_markup(session)
    original_spots = deepcopy(session.spots.saved_state())
    session.select_markup({"node": markup["id"], "annotation": item["id"]})
    source = np.zeros_like(session.rendered.image)
    assert np.any(session.draw_markup_handles(source)) and not np.any(source)

    def pointer(kind: str, x: int, y: int) -> None:
        """
        Send real validated image-local pointer messages rather than calling geometry directly.
        """
        session.viewer_pointer({"kind": kind, "button": "left", "x": x, "y": y})

    pointer("press", 77, 58)
    assert session.markup_gesture.handle == 0
    pointer("move", 97, 78)
    assert read_annotations(markup["params"]["annotations"])[0]["points"] == item["points"]
    pointer("release", 97, 78)
    moved = deepcopy(read_annotations(markup["params"]["annotations"])[0]["points"])
    assert moved[0] != item["points"][0] and moved[1] == item["points"][1]
    assert session.markup_selection is not None and session.markup_selection.index == 0
    viewer_panel = cast(Mock, session.viewer)
    viewer_panel.poll.return_value = [{"action": "measurement_nudge", "direction": "right"}]
    session.poll_viewer_commands()
    nudged = deepcopy(read_annotations(markup["params"]["annotations"])[0]["points"])
    assert nudged[0][0] - moved[0][0] == pytest.approx(1 / 767)
    assert nudged[0][1] == moved[0][1] and nudged[1] == moved[1]
    assert not session.spot_drag.edit.pending_nudges
    markup["params"]["locked"] = True
    session.markup_nudge("right")
    assert session.markup_selection is None
    assert read_annotations(markup["params"]["annotations"])[0]["points"] == nudged
    assert session.spots.saved_state() == original_spots


@pytest.mark.usefixtures("viewer")
@pytest.mark.parametrize("matrix", [IDENTITY, (-1., 0., 1., 0., 1., 0., 0., 0., 1.),
                                   (0., -1., 1., 1., 0., 0., 0., 0., 1.)])
def test_whole_object_nudge_follows_viewer_axes_after_transforms(matrix: tuple[float, ...]) -> None:
    """
    Screen-right always moves right, even when source coordinates mirror or rotate.
    """
    session = desktop.DesktopSession(desktop, [])
    markup, item = setup_markup(session, matrix)
    session.markup_selection = MarkupSelection(markup["id"], item["id"], matrix=matrix)
    before = project(item["points"], matrix)
    assert session.markup_nudge("right")
    changed = read_annotations(markup["params"]["annotations"])[0]
    after = project(changed["points"], matrix)
    np.testing.assert_allclose(after - before, np.array([[1 / 767, 0.], [1 / 767, 0.]]), atol=1e-10)
    assert session.markup_selection is not None and session.markup_selection.index == -1


@pytest.mark.usefixtures("viewer")
def test_selection_clears_on_bypass_and_cancelled_vertex_drag_does_not_save() -> None:
    """
    A removed canvas/target cannot receive nudges, and outside releases discard control edits.
    """
    session = desktop.DesktopSession(desktop, [])
    markup, item = setup_markup(session)
    session.select_markup({"node": markup["id"], "annotation": item["id"]})
    session.viewer_pointer({"kind": "press", "button": "left", "x": 77, "y": 58})
    session.viewer_pointer({"kind": "release", "button": "left", "x": -1, "y": -1})
    assert read_annotations(markup["params"]["annotations"])[0]["points"] == item["points"]
    markup["bypass"] = True
    assert session.selected_markup() is None and session.markup_selection is None
