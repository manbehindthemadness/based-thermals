"""
Share display palettes and bounded scalar validation with portable visualization packages.
"""

import math
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from ..palettes import PREFIX, read_palette
from .palettes import palette_key
from .palettes import palette_lut as app_palette_lut


class PaletteNames(tuple):
    """
    Retain legacy named options while accepting validated portable gradient selections.

    Existing embedded surface packages validate with ``value in palette_names()``.
    Keeping that contract avoids rewriting user-owned package source on load.
    """

    def __contains__(self, value: object) -> bool:
        """
        Recognize complete custom snapshots without enumerating user-library entries.
        """
        if isinstance(value, str) and value.startswith(PREFIX):
            read_palette(value)
            return True
        if isinstance(value, str):
            try:
                palette_key(value)
            except ValueError:
                return False
            return True
        return super().__contains__(value)


@lru_cache(maxsize=1)
def palette_names() -> tuple[str, ...]:
    """
    Expose every pipeline palette, retaining the original surface choices first.
    """
    from ..pipeline_model.catalog import PALETTES

    names = ["Inferno", "Turbo", "Grayscale"]
    for key, label in PALETTES.items():
        name = f"{label} (approx.)" if key.startswith("camera_") else label
        if name not in names:
            names.append(name)
    return PaletteNames(names)


@lru_cache(maxsize=32)
def palette_lut(name: str) -> np.ndarray:
    """
    Return an immutable 256×3 BGR ramp using the application's palette implementation.
    """
    result = app_palette_lut(name)
    return result


def number(
    config: dict[str, Any],
    key: str,
    default: float,
    low: float,
    high: float,
    *,
    integer: bool = False,
) -> float:
    """
    Validate a finite scalar before allocating visualization buffers.
    """
    value = config.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and int(value) != value)
    ):
        raise ValueError(
            f"{key} must be {'an integer' if integer else 'a number'} in {low}..{high}"
        )
    return float(value)


def selection(config: dict[str, Any], key: str, default: str, options: tuple[str, ...]) -> str:
    """
    Reject malformed or unsupported choice values in standalone callbacks as well as JSON.
    """
    value = config.get(key, default)
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Unsupported {key}: {value!r}")
    return value


def flag(config: dict[str, Any], key: str, default: bool) -> bool:
    """
    Require an actual boolean rather than accepting numeric truth values.
    """
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def grayscale(image: np.ndarray, smoothing: float) -> np.ndarray:
    """
    Read a finite, nonempty grayscale display plane without altering the source.
    """
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
        or not np.issubdtype(image.dtype, np.number)
        or not np.isfinite(image).all()
    ):
        raise ValueError("Thermal visualization needs a finite H×W×3 display image")
    if np.max(np.ptp(image, axis=2)) > 1:
        raise ValueError("Thermal visualization needs grayscale input: select Source Raw")
    heat = np.array(image[:, :, 0], dtype=np.float32, copy=True)
    if smoothing:
        heat = cv2.GaussianBlur(heat, (0, 0), smoothing, borderType=cv2.BORDER_REPLICATE)
    return heat


def normalized(heat: np.ndarray, mode: str) -> np.ndarray:
    """
    Normalize relative display brightness, never interpreting pixels as Celsius or Fahrenheit.
    """
    if mode == "Auto contrast":
        low, high = np.percentile(heat, (1, 99))
        result = (heat - float(low)) / max(float(high - low), 1.0)
    else:
        result = heat / 255.0
    return np.asarray(np.clip(result, 0, 1), dtype=np.float32)


def display_settings(config: dict[str, Any]) -> tuple[np.ndarray, str, float]:
    """
    Validate the common palette, normalization and smoothing controls.
    """
    colors = app_palette_lut(config.get("palette", "turbo"))
    mode = selection(config, "normalization", "Source range", ("Source range", "Auto contrast"))
    smoothing = number(config, "smoothing", 1, 0, 5)
    return colors, mode, smoothing
