"""
Restore saved environmental corrections with one verified camera block write.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from test_hardware_controls import controls

from based_thermals.camera_backends.duo.hardware_controls import HARDWARE_CONTROLS, CameraError

SETTINGS = {
    "ambient": 22.5, "reflected": 24.5, "distance": 0.5,
    "emissivity": 0.95, "humidity": 16.5, "transmission": 99,
}


def test_saved_environment_uses_one_write_and_preserves_complete_snapshot(controls: Any) -> None:
    """
    Combine six owned targets while preserving unrelated bytes and the original baseline.
    """
    original = controls.original.copy()
    controls.camera.device.blocks[3, 1][8] = 123
    controls.camera.device.writes.clear()
    controls.set_environment(SETTINGS)
    assert [key for key, _ in controls.camera.device.writes] == [(3, 1)]
    assert controls.camera.device.blocks[3, 1][8] == 123
    assert controls.original == original
    assert controls.enabled == SETTINGS.keys()
    for name, value in SETTINGS.items():
        assert controls.state()[name]["value"] == value
        assert HARDWARE_CONTROLS[name].decode(controls.camera.device.blocks[3, 1]) == value
    controls.set_environment(SETTINGS)
    assert len(controls.camera.device.writes) == 1
    controls.restore()
    expected = bytearray(original[3, 1])
    expected[8] = 123
    assert controls.camera.device.blocks[3, 1] == expected


def test_startup_environment_rejects_all_values_before_any_write(controls: Any) -> None:
    """
    Do not partially apply a saved configuration when a later target is invalid.
    """
    controls.write = Mock()
    with pytest.raises(ValueError):
        controls.set_environment({**SETTINGS, "emissivity": 1.2})
    controls.write.assert_not_called()
    assert not controls.enabled
    with pytest.raises(ValueError, match="environmental"):
        controls.set_environment({"brightness": 25})


def test_startup_environment_rolls_back_failed_readback_without_publishing_targets(
    controls: Any,
) -> None:
    """
    Preserve the existing transactional behavior when the camera ignores a block write.
    """
    previous = bytes(controls.camera.device.blocks[3, 1])
    controls.camera.device.ignore_next_write = True
    with pytest.raises(CameraError, match="did not apply"):
        controls.set_environment(SETTINGS)
    assert not controls.enabled
    assert bytes(controls.camera.device.blocks[3, 1]) == previous
    assert controls.camera.device.writes[-1] == ((3, 1), previous)


__all__ = ["controls"]
