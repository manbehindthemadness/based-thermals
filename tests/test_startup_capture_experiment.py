"""
Bound passive startup recording and preserve the existing camera owner and callbacks.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import experiment_startup_frames as experiment
from experiments.experiment_startup_frames import StartupCapture


def test_capture_samples_startup_without_altering_or_reordering_frames(tmp_path: Path) -> None:
    """
    Copy no pixel data during acquisition and write original samples only on explicit save.
    """
    now = [0.0]
    capture = StartupCapture(clock=lambda: now[0])
    source = [bytes((index,)) * 8 for index in range(120)]
    previous = Mock()

    def frames() -> Iterator[bytes]:
        """
        Simulate an existing backend emitting frames and mutable stream counters.
        """
        for index, frame in enumerate(source):
            camera.stream_observer({"frames": index})
            now[0] += 0.04
            yield frame

    camera = SimpleNamespace(frames=frames, stream_observer=previous, mode=None)
    assert capture.wrap(camera) is camera
    assert list(camera.frames()) == source
    assert camera.stream_observer is previous
    assert previous.call_count == len(source)
    sequences = [sequence for sequence, _, _ in capture.frames]
    assert sequences == [*range(1, 33), *range(36, 121, 4)]
    assert capture.frames[0][2] is source[0]
    directory = tmp_path / "evidence"
    assert not directory.exists()
    capture.save(directory)
    index = json.loads((directory / "index.json").read_text())
    assert index["received_frames"] == 120
    assert index["stored_bytes"] == sum(len(payload) for _, _, payload in capture.frames)
    assert (directory / "frame-0001.bin").read_bytes() == source[0]
    assert len(index["frames"]) == len(capture.frames)


def test_capture_limits_time_frames_bytes_and_stream_event_copies() -> None:
    """
    Stop collecting after the fixed bounds while allowing normal acquisition to continue.
    """
    now = [0.0]
    capture = StartupCapture(clock=lambda: now[0])
    event = {"rejected": {"partial": 1}}
    capture.event(event)
    event["rejected"]["partial"] = 99
    assert capture.events[0]["counters"]["rejected"]["partial"] == 1
    payload = bytes(1024 * 1024)
    for _ in range(500):
        capture.record(payload)
        capture.event(event)
    assert capture.stored_bytes == 64 * 1024 * 1024
    assert len(capture.frames) <= 96 and len(capture.events) == 128
    small = StartupCapture(clock=lambda: now[0])
    for _ in range(500):
        small.record(b"small")
    assert len(small.frames) == 96
    timed = StartupCapture(clock=lambda: now[0])
    timed.record(b"first")
    now[0] = 13
    timed.record(b"late")
    assert len(timed.frames) == 1


@pytest.mark.parametrize("failure", [False, True])
def test_launcher_uses_normal_owner_and_saves_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool,
) -> None:
    """
    Preserve normal teardown and factory state on both successful and failed viewer exits.
    """
    camera = SimpleNamespace(frames=lambda: iter([b"original"]), close=Mock(),
                             stream_observer=None, mode=None)
    factory = Mock(return_value=camera)
    api = experiment.desktop
    monkeypatch.setattr("based_thermals.desktop.create_camera", factory)

    def app(argv: list[str]) -> int:
        """
        Stand in for the normal viewer and its existing camera shutdown path.
        """
        assert argv == ["--rotate", "90"]
        owner = api.create_camera()
        try:
            assert list(owner.frames()) == [b"original"]
            assert not list(tmp_path.iterdir())
            if failure:
                raise RuntimeError("viewer failed")
            return 0
        finally:
            owner.close()

    monkeypatch.setattr("based_thermals.desktop.main", app)
    args = ["--output", str(tmp_path), "--", "--rotate", "90"]
    if failure:
        with pytest.raises(RuntimeError, match="viewer failed"):
            experiment.main(args)
    else:
        assert experiment.main(args) == 0
    assert api.create_camera is factory
    camera.close.assert_called_once()
    assert len(list(tmp_path.glob("*/frame-0001.bin"))) == 1
