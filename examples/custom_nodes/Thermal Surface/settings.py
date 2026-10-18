"""
Validate surface settings without probing devices or loading external resources.
"""

import math
from dataclasses import dataclass
from typing import Any


def number(config: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    """
    Require a finite number in the renderer's supported range.
    """
    value = config.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"Thermal Surface: {key} must be a number from {low} to {high}")
    return float(value)


def choice(config: dict[str, Any], key: str, default: str, options: tuple[str, ...]) -> str:
    """
    Reject unsupported choices before allocating render buffers.
    """
    value = config.get(key, default)
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Thermal Surface: unsupported {key}: {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """
    Hold validated settings, with geometry separated from frame-varying heights.
    """

    detail: int
    yaw: float
    tilt: float
    height: float
    smoothing: float
    palette: str
    normalization: str
    resolution: int
    wireframe: bool
    scene_rotation: float
    offset_x: float
    offset_y: float
    zoom: float

    @classmethod
    def read(cls, config: dict[str, Any]) -> "Settings":
        """
        Accept partial standalone configuration using the same defaults as the node fields.
        """
        detail = number(config, "detail", 40, 12, 96)
        if not detail.is_integer():
            raise ValueError("Thermal Surface: detail must be an integer")
        wireframe = config.get("wireframe", False)
        if not isinstance(wireframe, bool):
            raise TypeError("Thermal Surface: wireframe must be a boolean")
        result = cls(
            int(detail), number(config, "yaw", -35, -180, 180),
            number(config, "tilt", 50, 15, 85), number(config, "height", 0.7, 0, 2),
            number(config, "smoothing", 0.8, 0, 3),
            choice(config, "palette", "Inferno", ("Inferno", "Turbo", "Grayscale")),
            choice(config, "normalization", "Source range", ("Source range", "Auto contrast")),
            int(choice(config, "resolution", "640", ("384", "640", "960"))), wireframe,
            number(config, "scene_rotation", 0, -180, 180),
            number(config, "offset_x", 0, -100, 100),
            number(config, "offset_y", 0, -100, 100), number(config, "zoom", 1, 0.1, 5),
        )
        return result
