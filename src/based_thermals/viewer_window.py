"""
Qt main viewer in a separate process, with controls outside the image canvas.
"""

from __future__ import annotations

import math
import mmap
from collections import deque
from collections.abc import Callable
from time import monotonic, perf_counter

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QCursor, QImage, QKeyEvent, QResizeEvent
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QWidget

from . import APP_NAME
from .capture_window import run_window
from .markup_ui.actions import ObjectActions
from .viewer_panel import FRAME_CAPACITY
from .viewer_ui.canvas import ImageCanvas
from .viewer_ui.drawing_toolbar import DrawingToolbar
from .viewer_ui.orientation import OrientationDialog
from .viewer_ui.toolbar import ViewerToolbar


def validated_draw_time(value: object) -> float | None:
    """
    Accept finite nonnegative elapsed milliseconds from a frame descriptor.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        duration = float(value)
    except OverflowError:
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def averaged_timing(samples: deque[float | None], value: float | None) -> float | None:
    """
    Average available timings in the displayed-frame window, preserving unavailable values.
    """
    samples.append(value)
    available = [sample for sample in samples if sample is not None]
    return sum(available) / len(available) if value is not None else None


class ViewerWindow(QMainWindow):
    """
    Own window geometry, fullscreen and input; acquisition stays in the parent.
    """

    def __init__(self, send: Callable[[dict], None]) -> None:
        """
        Build a native toolbar and a resizable black image area.
        """
        super().__init__()
        self.send = send
        self._closing = False
        self._mapping: mmap.mmap | None = None
        self._path = ""
        self._windowed_size: list[int] | None = None
        self._geometry_pending = False
        self._frame_draw_ms: float | None = None
        self._frame_copy_ms = 0.0
        self._viewport_draw_ms: float | None = None
        self._captured_at: float | None = None
        self._capture_to_paint_ms: float | None = None
        self._frame_paint_pending = False
        self._draw_samples: deque[float | None] = deque(maxlen=10)
        self._total_samples: deque[float | None] = deque(maxlen=10)
        self._paint_times: deque[float] = deque(maxlen=25)
        self._paint_count = 0
        self._metrics_at = monotonic()
        self._status_text = "Waiting for camera…"
        self.canvas = ImageCanvas(send, self._viewport_painted)
        self.setCentralWidget(self.canvas)
        self.toolbar = ViewerToolbar(self._command, self)
        self.addToolBar(self.toolbar)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar.graph_toolbar)
        self.toolbar.graph_toolbar.hide()
        self.drawing_toolbar = DrawingToolbar(self._command, self)
        self.addToolBarBreak()
        self.addToolBar(self.drawing_toolbar)
        self._viewer_state: dict = {}
        self._markup_target: dict = {}
        self._markup_response = ""
        self.markup_actions = ObjectActions(
            self.canvas, self._markup_object,
            lambda: (bool(self._viewer_state.get("markup_selected")),
                     bool(self._viewer_state.get("markup_editable")) and not self.drawing_toolbar.active),
            lambda: self.drawing_toolbar.active,
        )
        self.orientation_dialog = OrientationDialog(self.send, self)
        self.setWindowTitle(APP_NAME)
        self.resize(1024, 800)
        self.statusBar().setToolTip(
            "10-frame average of viewport preparation and painting; "
            "excludes camera waits, saving and input."
        )
        counters = QWidget()
        counter_layout = QHBoxLayout(counters)
        counter_layout.setContentsMargins(0, 0, 0, 0)
        counter_layout.setSpacing(16)
        self.status_text = QLabel()
        self.draw_counter = QLabel("-- ms")
        self.draw_counter.setToolTip(self.statusBar().toolTip())
        self.draw_counter.setFixedWidth(self.draw_counter.fontMetrics().horizontalAdvance("99999.9 ms"))
        self.draw_counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.total_counter = QLabel("Total: -- ms")
        self.total_counter.setToolTip(
            "10-frame average from complete camera frame receipt to completed Qt painting, "
            "including processing and queues; "
            "excludes sensor exposure and physical display refresh."
        )
        for label in (self.status_text, self.draw_counter, self.total_counter):
            counter_layout.addWidget(label)
        self.statusBar().addWidget(counters)
        self._update_status()

    def _viewport_painted(self, elapsed_ms: float) -> None:
        """
        Average drawing cost and source-frame latency once per displayed frame.
        """
        if self._frame_paint_pending:
            draw_ms = (
                self._frame_draw_ms + self._frame_copy_ms + elapsed_ms
                if self._frame_draw_ms is not None else None
            )
            self._viewport_draw_ms = averaged_timing(self._draw_samples, draw_ms)
            now = monotonic()
            self._paint_times.append(now)
            self._paint_count += 1
            total_ms = (
                (now - self._captured_at) * 1000
                if self._captured_at is not None and self._captured_at <= now else None
            )
            self._capture_to_paint_ms = averaged_timing(self._total_samples, total_ms)
            self._frame_paint_pending = False
            if self._viewer_state.get("readout_metrics") and now - self._metrics_at >= .5:
                duration = self._paint_times[-1] - self._paint_times[0]
                fps = (len(self._paint_times) - 1) / duration if duration > 0 else None
                self.send({"action": "viewer_metrics", "fps": fps, "frames": self._paint_count,
                           "draw_ms": self._viewport_draw_ms, "total_ms": self._capture_to_paint_ms})
                self._metrics_at = now
        self._update_status()

    def _markup_object(self, operation: str, text: str) -> None:
        """
        Forward bounded object edits with the context menu's stable target identity.
        """
        self.send({"action": "markup_object", "operation": operation, "text": text,
                   **self._markup_target})

    def _markup_reply(self, response: object) -> None:
        """
        Apply each clipboard/menu response once while normal frame updates continue.
        """
        if not isinstance(response, dict):
            return
        serial = response.get("serial")
        if not isinstance(serial, str) or serial == self._markup_response:
            return
        self._markup_response = serial
        text = response.get("clipboard")
        if isinstance(text, str) and len(text) <= 70000:
            QApplication.clipboard().setText(text)
        target = response.get("menu")
        if isinstance(target, dict):
            menu_target = dict(target)
            self._markup_target = menu_target
            self.markup_actions.popup(QCursor.pos())
            assert self.markup_actions.menu is not None
            # Hide first, then triggered actions run; clear the target on the next event-loop turn.
            self.markup_actions.menu.aboutToHide.connect(
                lambda: QTimer.singleShot(0, menu_target.clear)
            )
        self.send({"action": "markup_ack", "serial": serial})

    def _update_status(self) -> None:
        """
        Keep status visible beside the draw and total counters with fixed spacing.
        """
        draw = f"{self._viewport_draw_ms:.1f}" if self._viewport_draw_ms is not None else "--"
        total = f"{self._capture_to_paint_ms:.1f}" if self._capture_to_paint_ms is not None else "--"
        self.status_text.setText(self._status_text)
        self.draw_counter.setText(f"{draw} ms")
        self.total_counter.setText(f"Total: {total} ms")

    def _command(self, message: dict) -> None:
        """
        Handle window-local fullscreen, forwarding other actions unchanged.
        """
        if self._closing:
            return
        if message.get("action") == "quit":
            self.begin_closing()
        if message.get("action") == "rotate":
            self.orientation_dialog.open_state(self._viewer_state)
            return
        if message.get("action") == "fullscreen":
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            self.send(message)
        self.canvas.setFocus()

    def _read_frame(self, frame: dict) -> None:
        """
        Copy a validated BGR frame before acknowledging reuse of its shared storage.
        """
        started = perf_counter()
        width, height, path = frame.get("width"), frame.get("height"), frame.get("path")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or not isinstance(path, str)
        ):
            raise TypeError("Invalid viewer frame descriptor")
        if min(width, height) <= 0 or width * height * 3 > FRAME_CAPACITY:
            raise ValueError("Invalid viewer frame dimensions")
        frame_path = path
        if frame_path != self._path:
            self._close_mapping()
            with open(frame_path, "rb") as stream:
                self._mapping = mmap.mmap(stream.fileno(), FRAME_CAPACITY, access=mmap.ACCESS_READ)
            self._path = frame_path
        assert self._mapping is not None
        pixels = self._mapping[: width * height * 3]
        self.canvas.image = QImage(
            pixels, width, height, width * 3, QImage.Format.Format_BGR888
        ).copy()
        self._frame_draw_ms = validated_draw_time(frame.get("viewport_draw_ms"))
        self._captured_at = validated_draw_time(frame.get("captured_at"))
        self._frame_paint_pending = True
        self._frame_copy_ms = (perf_counter() - started) * 1000 + (
            validated_draw_time(frame.get("viewport_copy_ms")) or 0.0
        )
        self.canvas.update()
        self.send({"action": "frame_read", "serial": frame.get("serial")})

    def update_state(self, state: dict) -> None:
        """
        Update controls and image independently, without raising the window on each frame.
        """
        if state.get("closing", False):
            self.begin_closing()
        if self._closing:
            return
        frame = state.get("frame")
        if isinstance(frame, dict):
            try:
                self._read_frame(frame)
            except (OSError, ValueError, TypeError) as exc:
                self.send({"action": "error", "message": str(exc)})
                self.send({"action": "frame_read", "serial": frame.get("serial")})
        if "unit" in state:
            self._viewer_state = state
            self._markup_reply(state.get("markup_response", {}))
            self.toolbar.update_state(state)
            self.orientation_dialog.update_locks(state)
            drawing = state.get("measurement_drawing", {})
            drawing_was_active = self.drawing_toolbar.active
            self.drawing_toolbar.update_state(drawing if isinstance(drawing, dict) else {})
            if self.drawing_toolbar.active and not drawing_was_active:
                self.raise_()
                self.activateWindow()
                self.canvas.setFocus()
            self._status_text = str(state.get("status", "Ready"))
            self._update_status()
        size = state.get("resize")
        if (
            isinstance(size, list)
            and len(size) == 2
            and all(type(n) is int and 0 < n <= 16384 for n in size)
            and not self.isFullScreen()
        ):
            chart_height = (
                0
                if self.toolbar.graph_toolbar.isHidden()
                else self.toolbar.graph_toolbar.sizeHint().height()
            )
            self.resize(
                size[0],
                size[1]
                + self.toolbar.sizeHint().height()
                + chart_height
                + self.statusBar().height(),
            )
        self._schedule_geometry()

    def _schedule_geometry(self) -> None:
        """
        Coalesce native resize events after Qt has laid out toolbar and canvas.
        """
        if not self._geometry_pending:
            self._geometry_pending = True
            QTimer.singleShot(0, self._send_geometry)

    def _send_geometry(self) -> None:
        """
        Report image-area dimensions and the last windowed size for persistence.
        """
        self._geometry_pending = False
        size = [self.canvas.width(), self.canvas.height()]
        if not self.isFullScreen():
            drawing_height = (
                self.drawing_toolbar.height() if self.drawing_toolbar.isVisible() else 0
            )
            self._windowed_size = [size[0], size[1] + drawing_height]
        self.send({"action": "geometry", "size": size, "windowed_size": self._windowed_size})

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Let Qt lay out the canvas and then report its dimensions.
        """
        super().resizeEvent(event)
        self._schedule_geometry()

    def changeEvent(self, event: QEvent) -> None:
        """
        Track native fullscreen changes, including the macOS green window button.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self, "toolbar"):
            self.toolbar.buttons["fullscreen"].setChecked(self.isFullScreen())
            self.toolbar.buttons["fullscreen"].setText(
                "Exit fullscreen" if self.isFullScreen() else "Fullscreen"
            )
            self._schedule_geometry()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Keep viewer shortcuts and let Escape leave fullscreen before requesting quit.
        """
        if self._closing:
            event.accept()
            return
        key = event.key()
        directions: dict[int, str] = {
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
        }
        if key in directions and self.canvas.hasFocus() and not self.drawing_toolbar.active:
            self.send({"action": "measurement_nudge", "direction": directions[key]})
            event.accept()
            return
        if self.drawing_toolbar.active and key in (
            Qt.Key.Key_Escape,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Backspace,
        ):
            action = (
                "draw_cancel"
                if key == Qt.Key.Key_Escape
                else "draw_undo"
                if key == Qt.Key.Key_Backspace
                else "draw_finish"
            )
            self._command({"action": action})
            event.accept()
            return
        if key == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
            return
        text = event.text().lower()
        if key == Qt.Key.Key_Escape or text == "q":
            self._command({"action": "quit"})
        elif text in ("o", "p", " ", "g", "l", "f", "v", "s", "c"):
            self.send({"action": "key", "key": ord(text)})
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _close_mapping(self) -> None:
        """
        Release borrowed file storage after the Qt-owned image has been copied.
        """
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self._path = ""

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Keep shutdown visible until the parent releases hardware and closes its pipe.
        """
        if not self.property("parent_disconnected"):
            self._send_geometry()
            self._command({"action": "quit"})
            event.ignore()
            return
        self._close_mapping()
        super().closeEvent(event)

    def begin_closing(self) -> None:
        """
        Freeze interaction while the parent finalizes recordings and restores hardware.
        """
        self._closing = True
        self.setWindowTitle(f"{APP_NAME} — Closing")
        self._status_text = "Closing — restoring camera settings. Keep the camera connected…"
        self._update_status()
        self.draw_counter.hide()
        self.total_counter.hide()
        self.toolbar.setEnabled(False)
        self.toolbar.graph_toolbar.setEnabled(False)
        self.drawing_toolbar.setEnabled(False)
        self.canvas.setEnabled(False)
        self.orientation_dialog.close()


def main() -> int:
    """
    Run the Qt viewer without the popup windows' always-on-top flag.
    """
    result = run_window(ViewerWindow, APP_NAME, stays_on_top=False, poll_ms=10)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
