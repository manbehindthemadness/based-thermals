"""
Restore selected SDK fields without replacing unrelated camera configuration.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hardware_controls import HardwareControl


def restored_fields(
    current: bytes, original: bytes, controls: Iterable[HardwareControl]
) -> bytes:
    """
    Copy original field bytes and associated flags into a current settings block.
    """
    target = bytearray(current)
    for control in controls:
        start = control.offset
        size = struct.calcsize("<" + control.fmt)
        target[start : start + size] = original[start : start + size]
        for offset, _ in control.extra:
            target[offset] = original[offset]
    return bytes(target)
