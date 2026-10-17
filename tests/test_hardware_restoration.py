"""
Block-level cleanup preserves unrelated settings and stops after a lost device.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from test_hardware_controls import Device

from based_thermals.camera import CameraError
from based_thermals.camera_backends.duo import hardware_controls as module
from based_thermals.hardware_controls import BLOCK_LENGTHS, HARDWARE_CONTROLS, HardwareControls


@pytest.fixture
def controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HardwareControls:
    """
    Capture a complete baseline using the fake audited SDK transport.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    camera = Mock(device=Device())
    owner = HardwareControls(camera)
    owner.load()
    return owner


def test_restore_writes_each_owned_block_once_and_restores_preset_last(
    controls: HardwareControls,
) -> None:
    """
    Restore all fields, flags and inactive defaults with four verified block writes.
    """
    device = controls.camera.device
    controls.enabled = set(HARDWARE_CONTROLS) - {"palette", "noise_general"}
    controls._pipeline_defaults = {"palette": 22, "noise_general": 100}
    for spec in HARDWARE_CONTROLS.values():
        spec.apply(device.blocks[spec.selector, spec.command], spec.maximum)
    for data in device.blocks.values():
        data[0] = 123
    order = []
    controls.restore_tone = Mock(side_effect=lambda: order.append("tone"))
    controls.restore_fixed_range = Mock(side_effect=lambda: order.append("fixed"))
    controls._processing_preset_owned = True
    controls._original_processing_preset = "soft"
    controls._apply_processing_preset = Mock(side_effect=lambda preset: order.append(preset))
    original_write = controls.write

    def write(key: tuple[int, int], payload: bytes) -> None:
        """
        Record restoration order while retaining real fake-device readback verification.
        """
        order.append(key)
        original_write(key, payload)

    controls.write = write
    controls.restore()
    assert order == ["tone", "fixed", *BLOCK_LENGTHS, "soft"]
    assert [key for key, _ in device.writes] == list(BLOCK_LENGTHS)
    for key, data in device.blocks.items():
        expected = bytearray(controls.original[key])
        expected[0] = 123
        assert data == expected
    assert not controls.enabled and not controls._pipeline_defaults
    assert not controls._processing_preset_owned


def test_restore_preserves_unowned_known_fields_and_skips_unchanged_blocks(
    controls: HardwareControls,
) -> None:
    """
    Releasing ambient must retain independently changed emissivity and avoid redundant writes.
    """
    device = controls.camera.device
    controls.enabled = {"ambient", "brightness"}
    HARDWARE_CONTROLS["ambient"].apply(device.blocks[3, 1], 23.5)
    HARDWARE_CONTROLS["emissivity"].apply(device.blocks[3, 1], 0.97)
    controls.restore()
    assert [key for key, _ in device.writes] == [(3, 1)]
    assert HARDWARE_CONTROLS["ambient"].decode(device.blocks[3, 1]) == 30
    assert HARDWARE_CONTROLS["emissivity"].decode(device.blocks[3, 1]) == 0.97
    assert not controls.enabled


@pytest.mark.parametrize("failure", ["read", "write"])
def test_failed_restore_stops_without_rollback_or_subsequent_commands(
    controls: HardwareControls, failure: str,
) -> None:
    """
    Keep failed ownership and the disk baseline without retrying a disconnected camera.
    """
    device = controls.camera.device
    controls.enabled = {"brightness", "contrast", "ambient"}
    controls._pipeline_defaults = {"contrast": 50}
    for name in controls.enabled:
        spec = HARDWARE_CONTROLS[name]
        spec.apply(device.blocks[spec.selector, spec.command], spec.maximum)
    method = getattr(controls, failure)

    def operation(*args: Any) -> Any:
        """
        Disconnect on the second owned block after the first block restores successfully.
        """
        key = tuple(args) if failure == "read" else args[0]
        if key == (2, 2):
            raise CameraError("camera disconnected")
        return method(*args)

    replacement = Mock(side_effect=operation)
    setattr(controls, failure, replacement)
    controls.restore_processing_preset = Mock()
    assert controls.snapshot_path is not None
    snapshot = controls.snapshot_path.read_bytes()
    with pytest.raises(CameraError, match="disconnected"):
        controls.restore()
    assert [key for key, _ in device.writes] == [(2, 1)]
    assert controls.enabled == {"contrast", "ambient"}
    assert controls._pipeline_defaults == {"contrast": 50}
    controls.restore_processing_preset.assert_not_called()
    assert controls.snapshot_path.read_bytes() == snapshot
