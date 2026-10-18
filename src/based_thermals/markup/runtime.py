"""
Small rolling counters for annotation readouts, without retaining frames or sample histories.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReadoutRuntime:
    """
    Keep one session clock, a bounded receipt-rate window and last published viewer metrics.
    """

    started: float | None = None
    receipts: deque[float] = field(default_factory=lambda: deque(maxlen=25))
    frames: int = 0
    viewer: dict[str, Any] = field(default_factory=dict)
    viewer_at: float | None = None
    metrics: bool = False

    def observe(self, now: float, captured_at: float | None) -> None:
        """
        Count a newly received frame once and initialize clocks using the session monotonic domain.
        """
        if self.started is None:
            self.started = now
        if captured_at is not None and (not self.receipts or captured_at > self.receipts[-1]):
            self.receipts.append(captured_at)
            self.frames += 1

    def values(self, now: float) -> dict[str, Any]:
        """
        Return host-observed rates and elapsed values; stalled receipt rates decay to zero.
        """
        elapsed = max(0., now - self.started) if self.started is not None else 0.
        seconds = int(elapsed)
        fps = None
        if len(self.receipts) > 1 and self.receipts[-1] > self.receipts[0]:
            fps = ((len(self.receipts) - 1) / (self.receipts[-1] - self.receipts[0])
                   if now - self.receipts[-1] < 1 else 0.)
        viewer = dict(self.viewer)
        if self.viewer_at is not None and now - self.viewer_at > 1:
            viewer["Display.FPS"] = 0.
        return {"Capture.FPS": fps, "Capture.Frames": self.frames,
                "Session.Elapsed": elapsed,
                "Session.Timer": f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}",
                **viewer}
