"""
Keep upright label centers beside their arrows when rotating the camera, including older saved labels.
"""

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.frames import make_frame
from support.pipeline import raw_pipeline

from based_thermals import desktop
from based_thermals.custom_nodes.labels import transform_labels
from based_thermals.markup.anchors import center_offset
from based_thermals.markup.labels import IDENTITY, MarkupLabel, node_labels
from based_thermals.markup.model import annotation, read_annotations, serialize
from based_thermals.markup.paint import draw_object, label_bounds, masks_for, project, text_layout
from based_thermals.pipeline import node, validate_pipeline


def mapping(rotation: int) -> tuple[float, ...]:
    """
    Obtain the real viewer transform for a stable annotation definition.
    """
    item = annotation("text")
    item["points"] = [[.3, .4]]
    markup = node("software", "markup", annotations=serialize([item]))
    transformed = transform_labels(node_labels(markup), rotation)[1]
    assert isinstance(transformed, MarkupLabel)
    return transformed.matrix


def rotate(point: np.ndarray, rotation: int) -> np.ndarray:
    """
    Supply an independent clockwise coordinate reference for known viewer quarter turns.
    """
    x, y = point
    return np.array({0: (x, y), 90: (1 - y, x), 180: (1 - x, 1 - y), 270: (y, 1 - x)}[rotation])


def bounds(shape: tuple[int, int], item: dict[str, Any], matrix: tuple[float, ...]) -> np.ndarray:
    """
    Measure the drawn text box without including raster cache padding or contrast halos.
    """
    lines, width, line_height = text_layout(item)
    return np.asarray(label_bounds(shape, item, matrix, (width + 12, line_height * len(lines) + 8)))


@pytest.mark.parametrize("initial", [0, 90, 180, 270])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_legacy_label_center_rotates_with_arrow_tail_without_moving_initial_layout(
    initial: int, rotation: int,
) -> None:
    """
    Reproduce the screenshot layout: a label above an arrow must move below it after a half turn.
    """
    item = annotation("text")
    item.pop("text_offset")
    item.update(points=[[.3, .4]], text="Hardware Upsample", font_size=11)
    original = deepcopy(item)
    first_shape = (480, 640) if initial % 180 == 0 else (640, 480)
    first_matrix = mapping(initial)
    before = np.full((*first_shape, 3), 50, np.uint8)
    draw_object(before, item, first_matrix)
    first = bounds(first_shape, item, first_matrix)
    center = (first[:2] + first[2:]) / 2 / (first_shape[1] - 1, first_shape[0] - 1)
    source_center = rotate(center, (360 - initial) % 360)
    item["text_offset"] = center_offset(first_shape, item, first_matrix)
    assert item["text_offset"] is not None
    after = np.full_like(before, 50)
    draw_object(after, item, first_matrix)
    np.testing.assert_array_equal(before, after)
    assert item["points"] == original["points"]
    source_tail = source_center + [.0, .06]
    shape = (480, 640) if rotation % 180 == 0 else (640, 480)
    rectangle = bounds(shape, item, mapping(rotation))
    actual_center = (rectangle[:2] + rectangle[2:]) / 2
    expected = rotate(source_center, rotation) * (shape[1] - 1, shape[0] - 1)
    np.testing.assert_allclose(actual_center, expected, atol=.51)
    tail = project([source_tail.tolist()], mapping(rotation))[0] * (shape[1] - 1, shape[0] - 1)
    expected_delta = (rotate(source_tail, rotation) - rotate(source_center, rotation)) * (shape[1] - 1, shape[0] - 1)
    np.testing.assert_allclose(tail - actual_center, expected_delta, atol=.51)
    restored = validate_pipeline(raw_pipeline(node("software", "markup", annotations=serialize([item]))))
    assert read_annotations(restored["software"][2]["params"]["annotations"])[0] == item


def test_new_labels_and_changing_readout_width_use_the_same_center_and_cache_identity() -> None:
    """
    New labels are centered at placement; text changes resize around that center without pivot drift.
    """
    item = annotation("text")
    item.update(points=[[.5, .5]], text="1")
    for rotation in (0, 90, 180, 270):
        shape = (480, 640) if rotation % 180 == 0 else (640, 480)
        for text in ("1", "Hardware Upsample"):
            item["text"] = text
            rectangle = bounds(shape, item, mapping(rotation))
            np.testing.assert_allclose((rectangle[:2] + rectangle[2:]) / 2,
                                       np.array((shape[1] - 1, shape[0] - 1)) / 2, atol=.51)
    centered = masks_for((480, 640), item, IDENTITY)
    shifted = masks_for((480, 640), {**item, "text_offset": [.1, .05]}, IDENTITY)
    assert shifted is not centered and shifted.x != centered.x


@pytest.mark.parametrize("offset", [None, [], [0], [0, 0, 0], [True, 0], [float("nan"), 0], [1e9, 0]])
def test_invalid_center_offsets_are_rejected(offset: object) -> None:
    """
    Preserve portable schema bounds without accepting malformed saved anchor metadata.
    """
    item = annotation("text")
    item["text_offset"] = offset
    with pytest.raises(ValueError):
        serialize([item])


@pytest.mark.usefixtures("viewer")
def test_existing_label_layout_gets_a_persisted_center_once_without_hardware_changes() -> None:
    """
    Initialize saved legacy labels from their current display, then reuse the center across restarts/rotations.
    """
    session = desktop.DesktopSession(desktop, [])
    item = annotation("text")
    item.pop("text_offset")
    item.update(points=[[.3, .4]], text="Raw Thermal Sensor", font_size=11)
    markup = node("software", "markup", annotations=serialize([item]), locked=True)
    session.pipeline = validate_pipeline(raw_pipeline(markup))
    session.renderer.set_pipeline(session.pipeline)
    session.rendered = session.api.replace(session.renderer.render_detailed(make_frame()),
                                           image=np.zeros((768, 576, 3), np.uint8))
    session.markup_frames = [{"node": markup["id"], "annotation": item["id"], "matrix": list(mapping(90))}]
    hardware = Mock()
    session.hardware.set = hardware
    session.anchor_legacy_markup_labels()
    saved = desktop.load_settings()["pipeline"]
    saved_node = next(n for n in saved["software"] if n["id"] == markup["id"])
    saved_item = read_annotations(saved_node["params"]["annotations"])[0]
    assert saved_item["points"] == item["points"] and "text_offset" in saved_item
    encoded = saved_node["params"]["annotations"]
    session.markup_frames[0]["matrix"] = list(mapping(270))
    session.anchor_legacy_markup_labels()
    current = session.markup_node(markup["id"])
    assert current is not None and current["params"]["annotations"] == encoded
    assert saved_node["params"]["locked"]
    hardware.assert_not_called()


def test_center_on_a_projective_horizon_does_not_crash_or_create_stray_text() -> None:
    """
    Skip an invisible center outside a valid source canvas instead of converting NaN/infinity to pixels.
    """
    item = annotation("text")
    item.update(points=[[.5, .5]], text_offset=[-1.5, 0.])
    perspective = (1., 0., 0., 0., 1., 0., 1., 0., 1.)
    assert not masks_for((480, 640), item, perspective).text.size
    singular = (0.,) * 9
    legacy = {key: value for key, value in item.items() if key != "text_offset"}
    assert center_offset((480, 640), legacy, singular) is None
