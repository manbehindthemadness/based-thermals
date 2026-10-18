"""Shutter pauses must not enter measurements, averaging or logs."""

import csv
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_camera import make_frame

from based_thermals.camera_backends.duo.camera import (
    IMAGE_OFFSET,
    SENSOR_PIXELS,
    YUY2_FRAME_BYTES,
    NegotiatedFrameAssembler,
)
from based_thermals.desktop import save_capture
from based_thermals.graphs import GraphSnapshot, GraphWorker
from based_thermals.render import ThermalRenderer
from based_thermals.web import LiveStream, create_app


def frozen_frame(value=20_000):
    frame = bytearray(make_frame(value))
    struct.pack_into("<I", frame, 32, 1)
    return bytes(frame)


@pytest.mark.parametrize("bad_frame", [frozen_frame(30_000), make_frame(0), make_frame(65535)])
def test_hold_and_recover_without_contaminating_average(bad_frame):
    renderer = ThermalRenderer(scale=1, smoothing=0.25)
    renderer.native_temperatures = True
    before = renderer.render_detailed(make_frame(4500))
    for _ in range(16):
        held = renderer.render_detailed(bad_frame)
        assert not held.measurements_valid
        assert held.measurement_status
        assert held.stats == before.stats
        np.testing.assert_array_equal(held.raw_counts, before.raw_counts)
        np.testing.assert_array_equal(held.temperatures_celsius, before.temperatures_celsius)
        np.testing.assert_array_equal(held.image, before.image)
    after = renderer.render_detailed(make_frame(4800))
    assert after.measurements_valid and not after.measurement_status
    assert after.stats.center == 25
    assert np.all(after.temperatures_celsius == 25)


def test_one_invalid_pixel_cannot_poison_whole_scene():
    renderer = ThermalRenderer(scale=1)
    renderer.native_temperatures = True
    before = renderer.render_detailed(make_frame(4500))
    damaged = bytearray(make_frame(4500))
    struct.pack_into("<H", damaged, 4640, 65535)
    after = renderer.render_detailed(bytes(damaged))
    assert "1 invalid pixels" in after.measurement_status
    assert after.stats == before.stats


def test_real_scene_jump_remains_valid():
    renderer = ThermalRenderer(scale=1, smoothing=1)
    renderer.native_temperatures = True
    renderer.render_detailed(make_frame(4000))
    changed = renderer.render_detailed(make_frame(5400))
    assert changed.measurements_valid
    assert changed.stats.center == 5400 / 64 - 50


def test_frozen_startup_capture_has_no_fake_temperature(tmp_path):
    renderer = ThermalRenderer(scale=1)
    rendered = renderer.render_detailed(frozen_frame())
    assert not rendered.measurements_valid
    assert np.isnan(rendered.temperatures_celsius).all()
    assert all(value is None for value in rendered.stats.as_dict().values())
    _, data_path, json_path = save_capture(rendered, tmp_path, 22, 0, selected_pixel=(1, 1))
    metadata = json.loads(json_path.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert not metadata["measurements_valid"]
    assert metadata["selected_pixel"]["temperature_celsius"] is None
    with np.load(data_path) as data:
        assert not data["measurements_valid"]
    live = renderer.render_detailed(make_frame())
    assert live.measurements_valid
    assert np.isfinite(live.temperatures_celsius).all()


def test_graphs_gap_and_csv_skips_frozen_samples(tmp_path):
    worker = GraphWorker()
    live = GraphSnapshot((18, 20, 24, 21), (((0, 0), 22),), (200, 200), "C")
    frozen = replace(live, measurements_valid=False)
    try:
        path = worker.start_logging(tmp_path / "shutter.csv")
        # No submit: exercise the same sampler/logging functions without racing its thread.
        with worker._condition:
            worker._sample(live, 1)
            worker._write_log(live, 1)
            worker._sample(frozen, 1.5)
            worker._write_log(frozen, 1.5)
            worker._sample(live, 2)
            worker._write_log(live, 2)
        assert np.isnan(worker._master[1][1]).all()
        assert np.isnan(worker._spots[(0, 0)][1][1]).all()
        assert worker._master[2][1] == live.stats
        worker.stop_logging()
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 10  # Two valid samples, five series each.
    finally:
        worker.close()


def test_web_exposes_freeze_status_and_null_startup_readings() -> None:
    """
    Verify web exposes freeze status and null startup readings.
    """
    stream = LiveStream()
    rendered = stream.renderer.render_detailed(frozen_frame())
    stream.stats = rendered.stats.as_dict()
    response = create_app(stream).test_client().get("/api/status")
    response_data = response.get_json()
    assert isinstance(response_data, dict)
    assert response_data["measurement_status"] == rendered.measurement_status
    assert response_data["stats"]["center"] is None


def test_incomplete_startup_frames_never_reach_averages_graphs_or_csv(tmp_path: Path) -> None:
    """
    Keep shifted preview bytes out of the entire measurement path before and after startup.
    """
    preview = np.full(SENSOR_PIXELS * 4, 0x8038, dtype="<u2").tobytes()
    first = make_frame(4500)[:IMAGE_OFFSET * 2] + preview
    second = make_frame(4800)[:IMAGE_OFFSET * 2] + preview
    damaged = first[:10_000] + first[17_802:]
    assembler = NegotiatedFrameAssembler(YUY2_FRAME_BYTES)
    renderer = ThermalRenderer(scale=1, smoothing=0.5)
    renderer.native_temperatures = True
    worker = GraphWorker()
    try:
        path = worker.start_logging(tmp_path / "startup.csv")
        for index, payload in enumerate((damaged, first, damaged, second)):
            frame = assembler.feed(bytes((2, 0x82 | (index % 2))) + payload)
            if frame is None:
                continue
            rendered = renderer.render_detailed(frame)
            stats = rendered.stats
            snapshot = GraphSnapshot(
                (stats.minimum, stats.average, stats.maximum, stats.center), (), (200, 200), "C",
            )
            with worker._condition:
                worker._sample(snapshot, index)
                worker._write_log(snapshot, index)
        assert len(worker._master) == 2
        expected = (4500 + 4800) / 2
        np.testing.assert_array_equal(renderer.averaged_raw_counts, np.full((192, 256), expected))
        assert all(max(stats) < 26 for _, stats in worker._master)
        worker.stop_logging()
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 8  # Two complete frames, four global measurement series each.
    finally:
        worker.close()
        renderer.pipeline_processor.close()
