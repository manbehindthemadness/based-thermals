"""Frame-linked viewport timing and compact status-bar display."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.frames import make_frame
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.capture_panel import CapturePanel
from based_thermals.viewer_panel import ViewerPanel


def test_render_timing_excludes_saving_and_input(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Accumulate render/overlay time without capture operations or interaction delays.
    """
    del viewer
    session = desktop.DesktopSession(desktop, [])
    clock = [0.0]

    def advance(seconds: float) -> None:
        """
        Advance a deterministic clock only while the simulated stage is running.
        """
        clock[0] += seconds

    monkeypatch.setattr(desktop.time, "perf_counter", lambda: clock[0])
    # Controller methods are inherited and are intentionally replaced at this test boundary.
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "poll_viewer_commands", lambda: True)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "receive_frame", lambda: None)
    for method in ("poll_spot_and_graph_commands", "poll_camera_commands", "poll_capture_commands"):
        monkeypatch.setattr(session, method, lambda: None)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "render_frame", lambda: advance(0.002))
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "finish_capture", lambda: advance(10))
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "draw_status", lambda: advance(0.003))

    def clicks() -> bool:
        """
        Model slow input handling outside the viewport drawing stages.
        """
        advance(20)
        return True

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "handle_clicks", clicks)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "present_and_handle_keys", lambda: True)
    assert session.tick()
    assert session.viewport_draw_ms == pytest.approx(5.0)


def test_draw_time_is_attached_to_the_published_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A pending frame retains its own timing even when newer state arrives.
    """
    messages: list[dict] = []
    monkeypatch.setattr(CapturePanel, "open", lambda _self, _state: None)
    monkeypatch.setattr(CapturePanel, "is_open", property(lambda _self: True))
    monkeypatch.setattr(CapturePanel, "update", lambda _self, state: messages.append(state))
    panel = ViewerPanel()
    panel.open({})
    try:
        image = np.zeros((10, 20, 3), np.uint8)
        panel.present(image, {"viewport_draw_ms": 12.5, "captured_at": 1.0})
        descriptor = messages[-1]["frame"]
        assert descriptor["viewport_draw_ms"] == 12.5
        assert descriptor["captured_at"] == 1.0
        assert descriptor["viewport_copy_ms"] >= 0
        panel.present(image, {"viewport_draw_ms": 90})
        assert "frame" not in messages[-1]
        assert descriptor["viewport_draw_ms"] == 12.5
        assert descriptor["captured_at"] == 1.0
    finally:
        panel.close()


def test_display_uses_processed_source_capture_time(viewer: Any) -> None:
    """
    Display the older processed image's origin rather than the newest camera reading's origin.
    """
    del viewer
    session = desktop.DesktopSession(desktop, [])
    worker = Mock()
    session.diagnostics = Mock()
    worker.latest_with_metadata.return_value = (
        (session.pipeline_revision, np.zeros((192, 256, 3), np.uint8), "raw", 10.0, ""),
        {"captured_at": 100.0},
    )
    session.pipeline_worker = worker
    session.frame = make_frame()
    session.fresh_frame = True
    session.frame_captured_at = 200.0
    try:
        session.render_frame()
        assert worker.submit.call_args.kwargs["captured_at"] == 200.0
        assert session.display_captured_at == 100.0
        assert session.viewer_state()["captured_at"] == 100.0
    finally:
        session.renderer.pipeline_processor.close()
        session.support_export.close()
        session.support_logging.close()


def test_status_bar_combines_frame_copy_paint_and_preparation(tmp_path: Path) -> None:
    """
    Show a numeric total beside status, preserving frame matching and unavailable values.
    """
    script = r"""
import tempfile
from PySide6.QtWidgets import QApplication
from based_thermals import viewer_window
from based_thermals.viewer_ui import canvas
from based_thermals.viewer_panel import FRAME_CAPACITY

app = QApplication([])
window = viewer_window.ViewerWindow(lambda _message: None)
window.show()
app.processEvents()
window.update_state({"unit": "C", "status": "Ready"})
assert f"{window.status_text.text()} | {window.draw_counter.text()}" == "Ready | -- ms"
assert window.total_counter.text() == "Total: -- ms"
for value in (None, True, "5", -1, float("nan"), float("inf"), 10**1000):
    assert viewer_window.validated_draw_time(value) is None
assert viewer_window.validated_draw_time(0) == 0
with tempfile.NamedTemporaryFile() as stream:
    stream.truncate(FRAME_CAPACITY)
    stream.write(bytes([10, 20, 30]) * 20 * 10)
    stream.flush()
    stamps = iter((1.0, 1.002))
    viewer_window.perf_counter = lambda: next(stamps)
    window.update_state({"frame": {"path": stream.name, "serial": 1,
                         "width": 20, "height": 10, "viewport_draw_ms": 5.0,
                         "viewport_copy_ms": 1.5, "captured_at": 4.95}})
    viewer_window.monotonic = lambda: 5.0
    stamps = iter((2.0, 2.003))
    canvas.perf_counter = lambda: next(stamps)
    window.canvas.repaint()
    assert f"{window.status_text.text()} | {window.draw_counter.text()}" == "Ready | 11.5 ms"
    assert window.total_counter.text() == "Total: 50.0 ms"
    window.update_state({"unit": "C", "status": "Recording", "viewport_draw_ms": 999, "captured_at": 0})
    assert f"{window.status_text.text()} | {window.draw_counter.text()}" == "Recording | 11.5 ms"
    assert window.total_counter.text() == "Total: 50.0 ms"
    viewer_window.monotonic = lambda: 100.0
    stamps = iter((2.0, 2.003))
    window.canvas.repaint()
    assert window.total_counter.text() == "Total: 50.0 ms"
    window.draw_counter.parentWidget().layout().activate()
    assert window.total_counter.x() - window.draw_counter.x() - window.draw_counter.width() == 16
    stamps = iter((3.0, 3.002))
    viewer_window.perf_counter = lambda: next(stamps)
    window.update_state({"frame": {"path": stream.name, "serial": 2,
                         "width": 20, "height": 10, "viewport_draw_ms": 20.0}})
    stamps = iter((4.0, 4.003))
    canvas.perf_counter = lambda: next(stamps)
    window.canvas.repaint()
    assert abs(window._viewport_draw_ms - 18.25) < 1e-6
    assert window.total_counter.text() == "Total: -- ms"
import time
viewer_window.perf_counter = time.perf_counter
canvas.perf_counter = time.perf_counter
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


def test_counters_average_ten_displayed_frames_without_repaint_samples(tmp_path: Path) -> None:
    """
    Exercise startup, rolling eviction and unavailable timing without counting repaints.
    """
    script = r"""
from PySide6.QtWidgets import QApplication
from based_thermals import viewer_window
app = QApplication([])
window = viewer_window.ViewerWindow(lambda _message: None)
viewer_window.monotonic = lambda: 100.0
for value in range(1, 13):
    window._frame_draw_ms = float(value)
    window._frame_copy_ms = 0.0
    window._captured_at = 100.0 - value / 1000
    window._frame_paint_pending = True
    window._viewport_painted(0.0)
    expected = sum(range(max(1, value - 9), value + 1)) / min(value, 10)
    assert abs(window._viewport_draw_ms - expected) < 1e-6
    assert abs(window._capture_to_paint_ms - expected) < 1e-6
    window._viewport_painted(1000.0)
    assert abs(window._viewport_draw_ms - expected) < 1e-6
    assert len(window._draw_samples) == min(value, 10)
assert window.draw_counter.text() == "7.5 ms"
assert window.total_counter.text() == "Total: 7.5 ms"
window._frame_draw_ms = None
window._captured_at = None
window._frame_paint_pending = True
window._viewport_painted(0.0)
assert window.draw_counter.text() == "-- ms"
assert window.total_counter.text() == "Total: -- ms"
window._frame_draw_ms = 14.0
window._captured_at = 99.986
window._frame_paint_pending = True
window._viewport_painted(0.0)
assert window.draw_counter.text() == "9.1 ms"
assert window.total_counter.text() == "Total: 9.1 ms"
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
