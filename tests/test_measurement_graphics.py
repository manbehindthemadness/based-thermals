"""
Visualization outputs hide sensor graphics while retaining native samples and Markup readouts.
"""

import json
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.pipeline import raw_pipeline
from test_measurement_transforms import gradient_frame

from based_thermals import desktop
from based_thermals.custom_nodes.metadata import embedded_metadata
from based_thermals.desktop_app.measurement_drawing import MeasurementDrawingController
from based_thermals.markup.expressions import render_item
from based_thermals.markup.labels import IDENTITY
from based_thermals.markup.model import annotation, serialize
from based_thermals.pipeline import node, validate_pipeline
from based_thermals.pipeline_model.measurement_graphics import (
    image_feed,
    measurement_graphics_visible,
)
from based_thermals.pipeline_presets import predefined_presets


def visualization() -> dict:
    """
    Make a dependency-free projected-view stand-in with literal display-only output metadata.
    """
    source = 'OUTPUT_KIND = "visualization"\ndef process(image, config):\n    return 255 - image\n'
    return node("software", "custom", name="Projected view", package=json.dumps({"__init__.py": source}))


@pytest.mark.parametrize("source", ['OUTPUT_KIND = "unknown"', 'OUTPUT_KIND = False',
                                   'OUTPUT_KIND = str("image")',
                                   'OUTPUT_KIND = "image"\nOUTPUT_KIND = "visualization"'])
def test_output_kind_is_literal_validated_metadata(source: str) -> None:
    """
    Reject malformed or computed declarations without executing package statements.
    """
    with pytest.raises(ValueError):
        embedded_metadata(source)


def test_metadata_and_saved_height_map_snapshots_need_no_import_or_preference_rewrite() -> None:
    """
    Recognize both released snapshots and newly declared packages, independent of editable node names.
    """
    preset = predefined_presets()["Thermal Height Map"]
    custom = next(n for n in preset["software"] if n["type"] == "custom")
    package = json.loads(custom["params"]["package"])
    source = package["__init__.py"]
    assert embedded_metadata(source)["OUTPUT_KIND"] == '"visualization"'
    assert not measurement_graphics_visible(preset)
    package["__init__.py"] = source.replace('OUTPUT_KIND = "visualization"\n\n', '')
    custom["params"].update(name="Renamed saved node", package=json.dumps(package))
    saved = deepcopy(preset)
    assert not measurement_graphics_visible(preset) and preset == saved
    assert validate_pipeline(preset) == preset
    assert image_feed(json.dumps({"__init__.py": 'raise RuntimeError("Never import")\nOUTPUT_KIND = "image"'}))
    assert not image_feed(json.dumps({"__init__.py": 'raise RuntimeError("Never import")\nOUTPUT_KIND = "visualization"'}))


def test_bypasses_idle_branches_masks_and_full_replacements_follow_output_contribution() -> None:
    """
    Only contributing A visualizations hide graphics; auxiliary tabs provide footprints only.
    """
    visual = visualization()
    document = raw_pipeline(visual)
    document["branches"]["B"] = raw_pipeline(visualization())["software"][:-1]
    assert not measurement_graphics_visible(document)
    visual["bypass"] = True
    assert measurement_graphics_visible(document)  # The visualization in idle B is irrelevant.
    masked = node("software", "combine", tab="raw", mask_source="B", mode="opacity")
    document["software"].insert(-1, masked)
    assert measurement_graphics_visible(document)
    masked["params"].update(tab="B", mask_source="none", opacity=0)
    assert measurement_graphics_visible(document)
    masked["params"]["opacity"] = .5
    assert measurement_graphics_visible(document)
    visual["bypass"] = False
    assert not measurement_graphics_visible(document)
    replacement = node("software", "combine", tab="raw", mode="replace")
    document["software"].insert(-1, replacement)
    assert measurement_graphics_visible(document)
    document["branches"]["C"] = raw_pipeline(node("software", "transform", width=35, height=35))["software"][:-1]
    replacement["params"]["tab"] = "C"
    assert not measurement_graphics_visible(document)  # A normal inset does not restore the surface's registration.
    document = raw_pipeline(node("software", "filter"), node("software", "markup"),
                            node("software", "custom"))
    assert measurement_graphics_visible(document)


@pytest.mark.parametrize("tab", ["B", "C", "D"])
@pytest.mark.parametrize("mode", ["replace", "opacity"])
@pytest.mark.parametrize("inset", [False, True])
def test_auxiliary_visualizations_never_hide_measurement_graphics(
    tab: str, mode: str, inset: bool,
) -> None:
    """
    Connected visualization images and insets in any auxiliary tab leave native graphics visible.
    """
    document = raw_pipeline(node("software", "combine", tab=tab, mode=mode))
    nodes = raw_pipeline(visualization())["software"][:-1]
    if inset:
        nodes.append(node("software", "transform", width=35, height=35))
    document["branches"][tab] = nodes
    assert measurement_graphics_visible(validate_pipeline(document))


@pytest.mark.usefixtures("viewer")
def test_hidden_graphics_keep_sampling_graphs_readouts_and_enabled_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Use the real renderer and desktop worker boundary; restore graphics automatically on bypass.
    """
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock(size=None, windowed_size=None, over_image=True)
    session.diagnostics = Mock()
    session.show_graph = True
    session.show_instructions = False
    session.spots.placing = True
    session.spots.add((40, 50))
    session.spots.placing = False
    number = session.spots.numbers[0]
    item = annotation("text")
    item.update(points=[[.7, .7]], text="Desk {{ [Desk] :.2f }}", bindings={"Desk": f"Measurement.{number}.Average"})
    visual = visualization()
    markup = node("software", "markup", annotations=serialize([item]))
    document = raw_pipeline(visual, markup)
    session.pipeline = validate_pipeline(document)
    session.renderer.set_pipeline(session.pipeline)
    frame = gradient_frame()
    session.frame = frame
    session.fresh_frame = True
    session.dynamic_markers = True
    saved_spots = deepcopy(session.spots.saved_state())
    worker = Mock()
    session.pipeline_worker = worker
    image = np.full((576, 768, 3), 90, np.uint8)
    result = (session.pipeline_revision, image, "raw", 1., "")
    worker.latest.return_value = result
    worker.latest_with_metadata.return_value = (result, {"markup": [
        {"node": markup["id"], "annotation": item["id"], "matrix": list(IDENTITY)},
    ]})
    session.render_frame()
    rendered = session.rendered
    assert rendered.measurements_valid and rendered.measurement_mapping_valid
    assert not rendered.measurement_graphics_visible
    assert session.markup_live["Measurement.Valid"]
    assert session.markup_live[f"Measurement.{number}.Average"] == pytest.approx(26.25)
    assert render_item(item, session.markup_live)["text"] == "Desk 26.25"
    annotated = rendered.image.copy()
    assert np.any(annotated != image)
    assert session.spots.graph_readings(rendered.temperatures_celsius)
    assert desktop.draw_sample_spots(image, rendered, session.spots, 3) is image
    assert desktop.draw_dynamic_markers(image, rendered) is image
    session.picker.x, session.picker.y = 120, 150
    picked, position = desktop.draw_picker(rendered, session.picker, 3)
    assert picked is rendered.image and position is None
    draft_draw = Mock(side_effect=AssertionError("Hidden draft drawn"))
    monkeypatch.setattr(MeasurementDrawingController, "draw_measurement_draft", draft_draw)
    session.display = rendered.image
    session.draw_status()
    assert np.array_equal(session.display, annotated)
    assert session.graph_spots
    assert session.spots.saved_state() == saved_spots and session.dynamic_markers
    assert session.spots_state()["mapping_available"]
    restored = deepcopy(session.pipeline)
    next(n for n in restored["software"] if n["type"] == "custom")["bypass"] = True
    session.renderer.set_pipeline(restored)
    visible = session.renderer.render_detailed(frame, image_processing=False)
    assert visible.measurement_graphics_visible
    assert np.any(desktop.draw_sample_spots(image, visible, session.spots, 3) != image)
    assert session.spots.saved_state() == saved_spots
