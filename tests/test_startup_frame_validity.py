"""
Keep untrusted thermal startup data out of the viewer and asynchronous image worker.
"""

import struct
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from support.frames import make_frame

from based_thermals import desktop
from based_thermals.camera_backends import CameraFrame, decode_frame
from based_thermals.desktop_app.session import DesktopSession


def frozen_frame() -> bytes:
    """
    Mark a complete Duo frame as frozen using the verified telemetry flag.
    """
    frame = bytearray(make_frame())
    struct.pack_into("<I", frame, 32, 1)
    return bytes(frame)


@pytest.fixture
def session(viewer: Any) -> Iterator[DesktopSession]:
    """
    Use real receive/render coordination with isolated preferences and fake external boundaries.
    """
    current = DesktopSession(desktop, [])
    assert current.hardware is viewer.hardware
    current.viewer = Mock(closed=False)
    current.diagnostics = Mock()
    worker = Mock()
    worker.latest.return_value = None
    worker.latest_with_metadata.return_value = (None, {})
    current.pipeline_worker = worker
    current.poll_viewer_commands = Mock(return_value=True)
    current.poll_maintenance_commands = Mock(return_value=True)
    current.hardware_setup_pending = False
    try:
        yield current
    finally:
        current.renderer.pipeline_processor.close()
        current.support_export.close()
        current.support_logging.close()


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("bad", [
    frozen_frame(), make_frame(0), make_frame(65535),
    replace(decode_frame(make_frame()), raw_counts=None),
], ids=["frozen", "zero-counts", "saturated-counts", "missing-radiometry"])
def test_invalid_startup_stays_waiting_then_first_valid_frame_enters_worker(
    session: DesktopSession, monkeypatch: pytest.MonkeyPatch,
    platform: str, bad: bytes | CameraFrame,
) -> None:
    """
    Never publish early invalid images, samples or capture timestamps on either platform.
    """
    monkeypatch.setattr(desktop.sys, "platform", platform)
    worker = cast(Mock, session.pipeline_worker)
    viewport = cast(Mock, session.viewer)
    for frame in (bad, None, bad):
        session.frame = frame
        assert session.tick()
        assert session.frame is None and session.last_frame is None
        assert session.last_frame_at is None and not session.fresh_frame
        assert session.renderer.last_valid_frame is None
        assert session.renderer.averaged_raw_counts is None
    worker.submit.assert_not_called()
    assert viewport.present.call_count == 3
    waiting = viewport.present.call_args.args[0]
    assert waiting.shape == (480, 640, 3) and np.any(waiting)
    good = make_frame(4500)
    session.frame = good
    assert session.receive_frame() is None
    session.render_frame()
    assert session.rendered.measurements_valid
    assert session.last_frame == good and session.last_frame_at is not None
    assert session.fresh_frame
    assert worker.submit.call_count == 1
    np.testing.assert_array_equal(
        decode_frame(worker.submit.call_args.args[0]).raw_counts,
        decode_frame(good).raw_counts,
    )


def test_invalid_frames_after_startup_still_hold_last_valid_image(session: DesktopSession) -> None:
    """
    Keep established calibration hold/recovery behavior once a real image has been displayed.
    """
    worker = cast(Mock, session.pipeline_worker)
    good = make_frame(4500)
    session.frame = good
    assert session.receive_frame() is None
    session.render_frame()
    session.frame = frozen_frame()
    assert session.receive_frame() is None
    session.render_frame()
    assert not session.rendered.measurements_valid
    assert session.rendered.measurement_status
    held = worker.submit.call_args.args[0]
    np.testing.assert_array_equal(decode_frame(held).raw_counts, decode_frame(good).raw_counts)


def test_duo_initializes_only_after_valid_frame_and_discards_pre_settings_samples(
    session: DesktopSession, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Wait for transport synchronization and never average the frame captured before restoration.
    """
    from types import SimpleNamespace

    initialize = Mock(return_value=True)
    monkeypatch.setattr(
        "based_thermals.desktop_app.controls.ControlsController.initialize_hardware", initialize,
    )
    monkeypatch.setattr(desktop.time, "monotonic", lambda: 10.0)
    session.hardware_setup_pending = True
    pump = SimpleNamespace(captured_at=9.0)
    session.frame_pump = cast(Any, pump)
    for frame in (None, frozen_frame(), make_frame(0)):
        session.frame = frame
        assert session.receive_frame() is True
    initialize.assert_not_called()
    session.frame = make_frame(4500)
    assert session.receive_frame() is True
    initialize.assert_called_once()
    assert session.frame is None and session.last_frame is None
    assert session.renderer.averaged_raw_counts is None
    session.frame = make_frame(4500)
    assert session.receive_frame() is True  # A queued pre-settings frame remains ineligible.
    pump.captured_at = 10.1
    session.frame = make_frame(4800)
    assert session.receive_frame() is None
    session.render_frame()
    assert session.rendered.measurements_valid
    np.testing.assert_array_equal(session.renderer.averaged_raw_counts, np.full((192, 256), 4800))


def test_preview_only_camera_does_not_wait_for_radiometry(session: DesktopSession) -> None:
    """
    Preserve the first image of non-radiometric backends.
    """
    profile = replace(session.camera_profile, id="preview-test", radiometry=False)
    frame = CameraFrame(profile, preview_bgr=np.full((192, 256, 3), 80, np.uint8))
    session.frame = frame
    assert session.receive_frame() is None
    assert session.last_frame is frame and session.fresh_frame


def test_independent_visible_stream_is_not_blocked_by_thermal_startup(
    session: DesktopSession,
) -> None:
    """
    Preserve independent visible-camera presentation while thermal readings are unavailable.
    """
    profile = replace(session.camera_profile, id="dual-test", visible=True)
    frame = replace(decode_frame(frozen_frame()), profile=profile)
    session.frame = frame
    assert session.receive_frame() is None
    assert session.last_frame is frame and session.fresh_frame


__all__ = ["viewer_fixture"]
