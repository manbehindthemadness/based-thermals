"""
Apply saved environmental corrections in one verified SDK block transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..errors import CameraError
from .restoration import restored_fields

if TYPE_CHECKING:
    from .hardware_controls import HardwareControls


def apply_environment(hardware: HardwareControls, settings: dict[str, object]) -> None:
    """
    Validate all targets, preserve unrelated bytes, and publish ownership only after readback.
    """
    from .hardware_controls import HARDWARE_CONTROLS

    if not settings:
        return
    fields = {name: spec for name, spec in HARDWARE_CONTROLS.items()
              if (spec.selector, spec.command) == (3, 1)}
    if set(settings) - fields.keys():
        raise ValueError("Only environmental corrections can share the startup settings block")
    hardware.load()
    key = (3, 1)
    for name, value in settings.items():
        fields[name].apply(bytearray(hardware.original[key]), value)
    previous = hardware.read(*key)
    target = bytearray(restored_fields(previous, hardware.original[key], fields.values()))
    enabled = hardware.enabled | settings.keys()
    values = {**hardware.values, **settings}
    for name, spec in fields.items():
        if name in enabled:
            spec.apply(target, values[name])
    if target != previous:
        try:
            hardware.write(key, bytes(target))
        except CameraError:
            hardware.write(key, previous)
            raise
    hardware.enabled = set(enabled)
    hardware.values.update({name: cast(float, value) for name, value in settings.items()})
