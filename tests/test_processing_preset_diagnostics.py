"""
Keep preset timeout evidence diagnostic-only without extra camera commands or retries.
"""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from based_thermals.camera_backends.duo import hardware_controls as module


@pytest.mark.parametrize("mode", [None, 1])
@pytest.mark.parametrize("debug", [False, True])
def test_preset_timeout_logs_request_kind_without_retrying(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    mode: int | None, debug: bool,
) -> None:
    """
    Distinguish failed baseline reads from writes while preserving bounded polling and errors.
    """
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def advance(seconds: float) -> None:
        """
        Advance the deterministic polling clock without waiting on USB or real time.
        """
        now[0] += seconds

    monkeypatch.setattr(module.time, "sleep", advance)
    device = Mock()

    def transfer(kind: int, _request: int, _value: int, _index: int, data: Any,
                 **_kwargs: Any) -> int | bytes:
        """
        Accept one existing request and report an empty native mailbox until timeout.
        """
        return len(data) if kind == 0x41 else bytes(4)

    device.ctrl_transfer.side_effect = transfer
    controls = module.HardwareControls(SimpleNamespace(device=device))
    caplog.set_level(logging.DEBUG if debug else logging.INFO, logger=module.__name__)
    with pytest.raises(module.CameraError, match="Processing preset command timed out"):
        # noinspection PyProtectedMember
        controls._processing_command(mode)
    writes = [call for call in device.ctrl_transfer.call_args_list if call.args[0] == 0x41]
    assert len(writes) == 1
    assert 2 <= now[0] < 2.1
    expected = "read bank" if mode is None else "set mode 1"
    assert (f"Processing preset request: {expected}" in caplog.text) == debug
    assert (f"Processing preset timeout: {expected}" in caplog.text) == debug
    if debug:
        assert "last_length=00000000" in caplog.text
