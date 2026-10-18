"""
Display annotation validation, cached compositing and branch geometry without camera access.
"""

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from support.frames import frame_with_preview
from support.pipeline import raw_pipeline

from based_thermals.markup import paint
from based_thermals.markup.interaction import MarkupGesture
from based_thermals.markup.labels import IDENTITY
from based_thermals.markup.model import KINDS, annotation, read_annotations, serialize
from based_thermals.pipeline import node, validate_pipeline
from based_thermals.processing.scheduler import PipelineProcessor


def placed(kind: str = "rectangle", **style: Any) -> dict[str, Any]:
    """
    Build independent, small display-space fixtures for each supported primitive.
    """
    item = annotation(kind)
    item["points"] = ([[.2, .2]] if kind == "text" else
                      [[.2, .2], [.5, .2], [.4, .5]] if kind == "polygon" else
                      [[.2, .2], [.5, .5]])
    item.update(style)
    return item


@pytest.mark.parametrize("kind", KINDS)
def test_objects_round_trip_and_draw_without_source_mutation(kind: str) -> None:
    """
    Persist every tool and composite cached display masks without changing input ownership.
    """
    item = placed(kind, border_color="#ff0000", text_color="#00ff00", fill_color="#0000ff")
    document = validate_pipeline(raw_pipeline(node("software", "markup", annotations=serialize([item]))))
    restored = validate_pipeline(json.loads(json.dumps(document)))
    assert restored == document
    source = np.full((240, 320, 3), 70, np.uint8)
    image = source.copy()
    paint.draw_object(image, item, IDENTITY)
    assert np.any(image != source)
    shape = (240, 320)
    masks = paint.masks_for(shape, item, IDENTITY)
    recolored = {**item, "border_color": "dynamic", "border_opacity": .2}
    assert paint.masks_for(shape, recolored, IDENTITY) is masks
    assert not masks.border.flags.writeable
    hit = (112, 84) if kind == "ellipse" else (64, 48)
    assert paint.hit_test(shape, item, IDENTITY, hit)
    assert not paint.hit_test(shape, item, IDENTITY, (315, 235))
    assert np.all(source == 70)


@pytest.mark.parametrize("change", [
    {"points": [[float("nan"), .2], [.5, .5]]}, {"points": [[-.1, .2], [.5, .5]]},
    {"border_opacity": 1.1}, {"fill": 1}, {"width": True}, {"font_size": 100},
    {"text": "x" * 1025}, {"border_color": "red"}, {"kind": "python"},
    {"unexpected": "field"}, {"points": [[.2, .2]]},
])
def test_invalid_annotations_are_rejected(change: dict[str, Any]) -> None:
    """
    Reject malformed portable configuration before it reaches rendering or interaction.
    """
    with pytest.raises((TypeError, ValueError)):
        serialize([placed(**change)])


def test_bounded_configuration_and_independent_opacity() -> None:
    """
    Enforce object/JSON budgets and independent fill/border alpha using fixed reference pixels.
    """
    with pytest.raises(ValueError):
        read_annotations("[]" + " " * 65536)
    with pytest.raises(ValueError):
        serialize([annotation("text") for _ in range(129)])
    item = placed(fill=True, fill_color="#ff0000", fill_opacity=.5, border_opacity=0)
    image = np.zeros((101, 101, 3), np.uint8)
    paint.draw_object(image, item, IDENTITY)
    assert image[35, 35].tolist() == [0, 0, 128]
    for background in (0, 127, 255):
        image.fill(background)
        paint.draw_object(image, placed(), IDENTITY)
        assert np.any(image != background)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_worker_deferral_transforms_and_native_frame_ownership(rotation: int) -> None:
    """
    Compare deferred final painting against scheduler painting after mirror and viewer rotation.
    """
    item = placed("arrow", border_color="#ff0000")
    markup = node("software", "markup", annotations=serialize([item]))
    document = raw_pipeline(markup, node("software", "mirror", horizontal=True))
    frame, _ = frame_with_preview()
    original = bytes(frame)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        expected, _ = processor.process(frame, None, document, scale=2, rotation=rotation)
        processor.defer_markup = True
        plain, _ = processor.process(frame, None, document, scale=2, rotation=rotation)
        assert not np.array_equal(plain, expected)
        entry = next(entry for entry in processor.last_markup if entry["annotation"] == item["id"])
        painted = plain.copy()
        paint.draw_object(painted, item, tuple(entry["matrix"]))
        np.testing.assert_array_equal(painted, expected)
        assert bytes(frame) == original
        markup["bypass"] = True
        processor.process(frame, None, document)
        assert processor.last_markup == []
    finally:
        processor.close()


def test_gesture_preview_commit_inverse_mapping_and_drag_clamping() -> None:
    """
    Keep previews transient, freeze released geometry and clamp translation in source coordinates.
    """
    item = placed()
    before = deepcopy(item)
    gesture = MarkupGesture()
    mirror = (-1., 0., 1., 0., 1., 0., 0., 0., 1.)
    gesture.begin("node", item, "draw", mirror)
    assert gesture.point((.8, .2)) == pytest.approx([.2, .2])
    gesture.pointer("press", [.1, .2])
    gesture.pointer("move", [.4, .6])
    assert not gesture.state()["finish"]
    gesture.pointer("release", [.4, .6])
    gesture.pointer("move", [.9, .9])
    assert gesture.geometry() == [[.1, .2], [.4, .6]]
    assert item == before
    gesture.begin("node", item, "drag", IDENTITY)
    gesture.pointer("press", [.3, .3])
    gesture.pointer("move", [1.2, 1.2])
    np.testing.assert_allclose(gesture.geometry(), [[.7, .7], [1., 1.]])
    gesture.cancel()
    assert not gesture.state()["active"]


def test_polygon_requires_committed_corners_and_cache_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Hover points cannot finish polygons; repeated geometry changes evict old raster memory.
    """
    gesture = MarkupGesture()
    gesture.begin("node", annotation("polygon"), "draw", IDENTITY)
    for point in ([.1, .1], [.4, .1], [.3, .4]):
        gesture.pointer("press", point)
        gesture.pointer("release", point)
    assert gesture.state()["finish"] and len(gesture.geometry()) == 3
    monkeypatch.setattr(paint, "MAX_BYTES", 100_000)
    for index in range(50):
        item = placed(width=index % 12 + 1, font_size=index % 80 + 8)
        paint.masks_for((240, 320), item, IDENTITY)
    assert paint._BYTES <= 100_000 and len(paint._CACHE) <= 256


@pytest.mark.parametrize("mode,opacity,expected", [
    ("opacity", .5, True), ("opacity", 0., False), ("mask", 1., False), ("replace", 1., True),
])
def test_connected_branch_transforms_and_mask_only_exclusion(
    mode: str, opacity: float, expected: bool,
) -> None:
    """
    Carry source-local geometry through a scaled inset while excluding mask-only inputs.
    """
    item = placed()
    markup = node("software", "markup", annotations=serialize([item]))
    document = raw_pipeline(node("software", "combine", tab="B", mode=mode, opacity=opacity))
    document["branches"]["B"] = [node("software", "source", source="raw"), markup,
                                 node("software", "transform", width=50, height=50)]
    frame, _ = frame_with_preview()
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        processor.process(frame, None, document)
        entries = [entry for entry in processor.last_markup if entry["annotation"] == item["id"]]
        assert bool(entries) == expected
        if entries:
            points = paint.project(item["points"], tuple(entries[0]["matrix"]))
            np.testing.assert_allclose(points, [[.1, .1], [.25, .25]], atol=.002)
        processor.process(frame, None, document)
        assert len(processor.last_markup) == (2 if expected else 0)
    finally:
        processor.close()


def test_legacy_z_defaults_and_validation() -> None:
    """
    Preserve existing Markup list order while accepting only bounded integer layer indices.
    """
    legacy = [placed(), placed()]
    for item in legacy:
        del item["z_index"]
    restored = read_annotations(json.dumps(legacy))
    assert [item["id"] for item in restored] == [item["id"] for item in legacy]
    assert [item["z_index"] for item in restored] == [0, 0]
    for value in (-1001, 1001, True, 1.5, None):
        with pytest.raises(ValueError, match="Z index"):
            serialize([placed(z_index=value)])


def test_scheduler_annotations_follow_z_across_nodes_and_stable_ties() -> None:
    """
    Higher layers win even when earlier in pipeline order; ties retain the original order.
    """
    front = placed(fill=True, fill_opacity=1., fill_color="#ff0000", z_index=4)
    back = placed(fill=True, fill_opacity=1., fill_color="#0000ff", z_index=-2)
    first = node("software", "markup", annotations=serialize([front]))
    second = node("software", "markup", annotations=serialize([back]))
    frame, _ = frame_with_preview()
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        image, _ = processor.process(frame, None, raw_pipeline(first, second))
        y, x = round((image.shape[0] - 1) * .35), round((image.shape[1] - 1) * .35)
        assert image[y, x].tolist() == [0, 0, 255]
        back["z_index"] = 4
        second["params"]["annotations"] = serialize([back])
        image, _ = processor.process(frame, None, raw_pipeline(first, second))
        assert image[y, x].tolist() == [255, 0, 0]
    finally:
        processor.close()
