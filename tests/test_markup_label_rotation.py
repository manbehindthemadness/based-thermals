"""
Upright labels retain visible text through camera rotation without changing saved anchor geometry.
"""

from copy import deepcopy

import cv2
import numpy as np
import pytest
from support.frames import frame_with_preview
from support.pipeline import raw_pipeline

from based_thermals.custom_nodes.labels import transform_labels
from based_thermals.markup.labels import IDENTITY, MarkupLabel, node_labels
from based_thermals.markup.model import annotation, serialize
from based_thermals.markup.paint import draw_object, hit_test, masks_for, project
from based_thermals.pipeline import node
from based_thermals.processing.scheduler import PipelineProcessor


def label(text: str = "Rotating label") -> dict:
    """
    Place an upright label near the lower-right corner where a single-point anchor can clip text.
    """
    item = annotation("text")
    item.update(points=[[.98, .97]], text=text, font_size=16)
    return item


def reference_glyphs(text: str, font_size: int) -> np.ndarray:
    """
    Draw an independent unclipped OpenCV reference at a safe fixed origin.
    """
    image = np.zeros((200, 400), np.uint8)
    for index, line in enumerate(text.splitlines()):
        cv2.putText(image, line, (20, 40 + index * round(font_size * 1.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 22, 255, 1, cv2.LINE_AA)
    return image


@pytest.mark.parametrize("rotation,expected", [
    (0, [.98, .97]), (90, [.03, .98]), (180, [.02, .03]), (270, [.97, .02]),
])
@pytest.mark.parametrize("text", ["Rotating label", "Stage one\nStage two"])
def test_complete_labels_stay_visible_without_moving_the_rotated_anchor(
    rotation: int, expected: list[float], text: str,
) -> None:
    """
    Compare every glyph with an unclipped reference and retain the known clockwise anchor mapping.
    """
    item = label(text)
    original = deepcopy(item)
    markup = node("software", "markup", annotations=serialize([item]))
    placed = transform_labels(node_labels(markup), rotation)[1]
    assert isinstance(placed, MarkupLabel)
    shape = (480, 640) if rotation % 180 == 0 else (640, 480)
    np.testing.assert_allclose(project(item["points"], placed.matrix)[0], expected, atol=1e-12)
    masks = masks_for(shape, item, placed.matrix)
    glyphs = reference_glyphs(text, item["font_size"])
    assert np.count_nonzero(masks.text) == np.count_nonzero(glyphs)
    assert int(masks.text.sum()) == int(glyphs.sum())
    assert np.any(masks.text)
    ys, xs = np.where(masks.text)
    point = int(xs[len(xs) // 2] + masks.x), int(ys[len(ys) // 2] + masks.y)
    assert hit_test(shape, item, placed.matrix, point)
    assert item == original


def test_interior_labels_keep_their_positions_and_off_canvas_labels_are_not_rescued() -> None:
    """
    Fit only in-frame anchors, retaining ordinary placement and intentional transformed cropping.
    """
    item = label()
    item.pop("text_offset")
    item["points"] = [[.2, .3]]
    masks = masks_for((480, 640), item, IDENTITY)
    assert masks.x == round(.2 * 639) - 18
    assert masks.y == round(.3 * 479) - 18
    outside = (1., 0., 1., 0., 1., 0., 0., 0., 1.)
    assert masks_for((480, 640), item, outside).border.size == 0


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_scheduler_and_live_deferral_preserve_label_visibility_and_round_trip(rotation: int) -> None:
    """
    Cover actual worker metadata, a connected mirrored branch and deferred drawing on rotation.
    """
    item = label()
    markup = node("software", "markup", annotations=serialize([item]))
    document = raw_pipeline(node("software", "combine", tab="B", mode="replace"))
    document["branches"]["B"] = [node("software", "source", source="raw"), markup,
                                 node("software", "transform", horizontal=True)]
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        painted, _ = processor.process(frame, None, document, scale=2, rotation=rotation)
        processor.defer_markup = True
        plain, _ = processor.process(frame, None, document, scale=2, rotation=rotation)
        entry = next(entry for entry in processor.last_markup if entry["annotation"] == item["id"])
        matrix = tuple(entry["matrix"])
        masks = masks_for(plain.shape[:2], item, matrix)
        assert int(masks.text.sum()) == int(reference_glyphs(item["text"], item["font_size"]).sum())
        deferred = plain.copy()
        draw_object(deferred, item, matrix)
        np.testing.assert_array_equal(painted, deferred)
        assert document == original
        processor.process(frame, None, document, scale=2, rotation=0)
        restored = next(entry for entry in processor.last_markup if entry["annotation"] == item["id"])
        np.testing.assert_allclose(project(item["points"], tuple(restored["matrix"])), [[.02, .97]])
    finally:
        processor.close()
