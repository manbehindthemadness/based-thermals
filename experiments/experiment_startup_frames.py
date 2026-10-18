"""
Record a bounded startup sample through the normal desktop camera owner for offline inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from based_thermals import desktop
from based_thermals.camera import NegotiatedMode
from based_thermals.camera_backends import CameraFrame


class StartupCapture:
    """
    Retain immutable frames in bounded memory; write files only after normal app cleanup.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """
        Limit collection to twelve seconds, 96 frames, 64 MiB and 128 stream events.
        """
        self.clock = clock
        self.started: float | None = None
        self.started_utc: str | None = None
        self.received = 0
        self.stored_bytes = 0
        self.frames: list[tuple[int, float, bytes]] = []
        self.events: list[dict[str, Any]] = []
        self.mode: dict[str, Any] | None = None
        self.lock = threading.Lock()

    def elapsed(self) -> float:
        """
        Start the collection window at the first observed frame or stream event.
        """
        now = self.clock()
        if self.started is None:
            self.started = now
            self.started_utc = datetime.now(UTC).isoformat()
        return now - self.started

    def record(self, frame: bytes | CameraFrame) -> None:
        """
        Keep the first 32 frames densely, then every fourth frame through the startup window.
        """
        payload = frame if isinstance(frame, bytes) else frame.source_bytes
        if payload is None:
            return
        with self.lock:
            elapsed = self.elapsed()
            self.received += 1
            if (elapsed > 12 or len(self.frames) >= 96
                    or self.stored_bytes + len(payload) > 64 * 1024 * 1024
                    or self.received > 32 and self.received % 4):
                return
            self.frames.append((self.received, elapsed, payload))
            self.stored_bytes += len(payload)

    def event(self, counters: dict[str, Any]) -> None:
        """
        Retain a bounded copy of existing USB stream counters without changing callbacks.
        """
        with self.lock:
            elapsed = self.elapsed()
            if elapsed <= 12 and len(self.events) < 128:
                self.events.append({"seconds": elapsed, "counters": deepcopy(counters)})

    def wrap(self, camera: Any) -> Any:
        """
        Observe the existing camera's frame iterator and diagnostics without opening another device.
        """
        original_frames = camera.frames

        def frames() -> Iterator[bytes | CameraFrame]:
            """
            Retain original frame bytes and continue yielding them unchanged to acquisition.
            """
            mode = getattr(camera, "mode", None)
            if isinstance(mode, NegotiatedMode):
                self.mode = asdict(mode)
            previous = getattr(camera, "stream_observer", None)

            def stream(counters: dict[str, Any]) -> None:
                """
                Copy stream health and preserve the application's original diagnostics observer.
                """
                self.event(counters)
                if callable(previous):
                    callback = cast(Callable[[dict[str, Any]], None], previous)
                    callback(counters)

            if hasattr(camera, "stream_observer"):
                camera.stream_observer = stream
            try:
                for frame in original_frames():
                    self.record(frame)
                    yield frame
            finally:
                if hasattr(camera, "stream_observer"):
                    camera.stream_observer = previous

        camera.frames = frames
        return camera

    def save(self, directory: Path) -> None:
        """
        Save original frame bytes and an index in a new directory after USB has been released.
        """
        directory.mkdir(parents=True, exist_ok=False)
        with self.lock:
            frames, events = list(self.frames), deepcopy(self.events)
        index = []
        for sequence, elapsed, payload in frames:
            name = f"frame-{sequence:04d}.bin"
            (directory / name).write_bytes(payload)
            index.append({"file": name, "sequence": sequence,
                          "seconds": round(elapsed, 6), "bytes": len(payload)})
        (directory / "index.json").write_text(json.dumps({
            "platform": sys.platform, "started_utc": self.started_utc,
            "mode": self.mode, "frames": index, "stream_events": events,
            "received_frames": self.received, "stored_bytes": self.stored_bytes,
            "boundary": "Assembled frames delivered by the normal camera backend, before decoding",
        }, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """
    Launch the normal app and save the bounded sample after its existing shutdown/restoration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/startup-frames"))
    args, viewer_args = parser.parse_known_args(argv)
    if viewer_args[:1] == ["--"]:
        viewer_args = viewer_args[1:]
    directory = args.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    capture = StartupCapture()
    original_factory = desktop.create_camera

    def factory(*factory_args: Any, **kwargs: Any) -> Any:
        """
        Wrap the same factory and USB helper the desktop normally uses.
        """
        camera = original_factory(*factory_args, **kwargs)
        return capture.wrap(camera)

    print(f"Startup trace: {directory.resolve()} — let the image settle, then quit normally.")
    desktop.create_camera = factory
    try:
        result = desktop.main(viewer_args)
    finally:
        desktop.create_camera = original_factory
        capture.save(directory)
        print(
            f"Saved {len(capture.frames)} startup frames "
            f"({capture.stored_bytes:,} bytes) to {directory}"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
