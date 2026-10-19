# Portable baseline snapshot: embedded Custom packages cannot import sibling example folders.
# noinspection DuplicatedCode
"""
Validate surface settings without probing devices or loading external resources.
"""

import math
from dataclasses import dataclass
from typing import Any


# noinspection DuplicatedCode
def number(config: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    """
    Require a finite number in the renderer's supported range.
    """
    value = config.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"Thermal Surface: {key} must be a number from {low} to {high}")
    return float(value)


# noinspection DuplicatedCode
def choice(config: dict[str, Any], key: str, default: str, options: tuple[str, ...]) -> str:
    """
    Reject unsupported choices before allocating render buffers.
    """
    value = config.get(key, default)
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Thermal Surface: unsupported {key}: {value!r}")
    return value


# noinspection DuplicatedCode
@dataclass(frozen=True)
class Settings:
    """
    Hold validated settings, with geometry separated from frame-varying heights.
    """

    execution: str
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
    antialiasing: str
    geometry_method: str

    @classmethod
    def read(cls, config: dict[str, Any]) -> "Settings":
        """
        Accept partial standalone configuration using the same defaults as the node fields.
        """
        detail = number(config, "detail", 256, 12, 512)
        if not detail.is_integer():
            raise ValueError("Thermal Surface: detail must be an integer")
        wireframe = config.get("wireframe", False)
        if not isinstance(wireframe, bool):
            raise TypeError("Thermal Surface: wireframe must be a boolean")
        result = cls(
            choice(config, "execution", "Taichi CPU", ("OpenCV", "Taichi CPU", "Taichi Metal",
                                                   "Taichi Vulkan", "Taichi CUDA")),
            int(detail), number(config, "yaw", 80, -180, 180),
            number(config, "tilt", 45, 15, 85), number(config, "height", 0.7, 0, 2),
            number(config, "smoothing", 3, 0, 3),
            choice(config, "palette", "Turbo", ("Inferno", "Turbo", "Grayscale")),
            choice(config, "normalization", "Source range", ("Source range", "Auto contrast")),
            int(choice(config, "resolution", "1280", ("384", "640", "960", "1280", "1600",
                                                     "1920", "2304", "2560", "3840", "4096"))),
            wireframe,
            number(config, "scene_rotation", 90, -180, 180),
            number(config, "offset_x", -14, -100, 100),
            number(config, "offset_y", 0, -100, 100), number(config, "zoom", 1, 0.1, 5),
            choice(config, "antialiasing", "2x supersampling", ("Off", "2x supersampling")),
            choice(config, "geometry_method", "NumPy", ("NumPy", "Taichi")),
        )
        return result
