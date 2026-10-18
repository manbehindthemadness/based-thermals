"""
Live annotation snapshots share native measurement scopes and avoid camera/control side effects.
"""

from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.frames import make_frame
from support.pipeline import raw_pipeline

from based_thermals import desktop
from based_thermals.markup.expressions import render_item
from based_thermals.markup.model import annotation, read_annotations, serialize
from based_thermals.measurements.model import MeasurementRegions
from based_thermals.pipeline import node
from based_thermals.processing.scheduler import PipelineProcessor


@pytest.mark.usefixtures("viewer")
def test_live_values_share_measurement_scope_and_hold_no_invalid_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Reuse readings per explicit frame, convert display units only and make invalid/disabled data unavailable.
    """
    session = desktop.DesktopSession(desktop, [])
    session.rendered = session.renderer.render_detailed(make_frame())
    session.spots.placing = True
    session.spots.add((40, 50))
    number = session.spots.numbers[0]
    item = annotation("text")
    item.update(text="{{ [Desk] - [Temperature.Minimum] :.1f }}",
                bindings={"Desk": f"Measurement.{number}.Average"}, points=[[.1, .1]])
    markup = node("software", "markup", annotations=serialize([item]))
    session.pipeline["software"].insert(1, markup)
    original = session.rendered.temperatures_celsius.copy()
    saved = deepcopy(session.spots.saved_state())
    sampling = session.spots.readings
    sampled = Mock(wraps=sampling)
    monkeypatch.setattr(MeasurementRegions, "readings", lambda _spots, temperatures: sampled(temperatures))
    session.spots.display_cache.begin_frame()
    session.update_markup_values()
    first = session.markup_live[f"Measurement.{number}.Average"]
    assert first == pytest.approx(float(original[50, 40]))
    assert render_item(item, session.markup_live)["text"] == "0.0"
    session.spots.graph_readings(session.rendered.temperatures_celsius)
    session.update_markup_values()
    assert sampled.call_count == 1
    session.renderer.temperature_unit = "F"
    session.update_markup_values()
    assert session.markup_live[f"Measurement.{number}.Average"] == pytest.approx(first * 1.8 + 32)
    assert np.array_equal(original, session.rendered.temperatures_celsius)
    session.spots.display_cache.end_frame()
    session.rendered = session.api.replace(session.rendered, measurements_valid=False)
    session.update_markup_values()
    assert session.markup_live["Measurement.Valid"] is False
    assert session.markup_live["Temperature.Minimum"] is None
    assert render_item(item, session.markup_live)["text"] == "--" and sampled.call_count == 1
    assert session.spots.saved_state() == saved
    hardware = cast(Mock, session.hardware)
    hardware.set.assert_not_called()
    assert read_annotations(markup["params"]["annotations"])[0]["text"] == item["text"]


@pytest.mark.usefixtures("viewer")
def test_snapshot_without_templates_does_not_sample_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ordinary viewer frames must not pay for another measurement reduction just for variable discovery.
    """
    session = desktop.DesktopSession(desktop, [])
    session.rendered = session.renderer.render_detailed(make_frame())
    sampled = Mock(side_effect=AssertionError("Unexpected sampling"))
    monkeypatch.setattr(MeasurementRegions, "readings", lambda _spots, temperatures: sampled(temperatures))
    session.update_markup_values()
    assert session.markup_live["Temperature.Unit"] == "C"
    assert session.markup_live["Session.Elapsed"] >= 0
    assert session.markup_live["Recording.Elapsed"] == 0
    sampled.assert_not_called()


@pytest.mark.usefixtures("viewer")
def test_metrics_are_validated_and_throttled_snapshots_remain_json_safe() -> None:
    """
    Ignore malformed, non-finite or excessive viewer telemetry instead of crashing the parent.
    """
    session = desktop.DesktopSession(desktop, [])
    session.markup_metrics({"fps": 24.5, "frames": 120, "draw_ms": 4., "total_ms": 35.})
    assert session.markup_runtime.viewer["Display.FPS"] == 24.5
    for bad in (True, -1, float("nan"), float("inf"), 10 ** 1000, "25"):
        session.markup_metrics({"fps": bad})
        assert session.markup_runtime.viewer["Display.FPS"] == 24.5


def test_direct_processor_and_previews_resolve_pipeline_settings_without_running_code() -> None:
    """
    Retain the same setting identity and fresh rendered text in actual deferred label processing.
    """
    gamma = node("software", "gamma", amount=1.2)
    item = annotation("text")
    key = f"Pipeline/{gamma['id']}/params/amount"
    item.update(points=[[.1, .1]], text="Gamma {{ [Gain] :.1f }}", bindings={"Gain": key})
    markup = node("software", "markup", annotations=serialize([item]))
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        document = raw_pipeline(gamma, markup)
        image, _ = processor.process(make_frame(), None, document)
        labels = processor.branches["A"].custom.labels
        definition = labels[-1].definition
        assert definition["text"] == "Gamma 1.2"
        gamma["params"]["amount"] = 2.1
        updated, _ = processor.process(make_frame(), None, document)
        assert processor.branches["A"].custom.labels[-1].definition["text"] == "Gamma 2.1"
        assert not np.array_equal(image, updated)
        assert read_annotations(markup["params"]["annotations"])[0]["text"] == item["text"]
    finally:
        processor.close()
