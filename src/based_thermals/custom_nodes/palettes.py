"""
Public app-palette access for Custom packages without embedded catalogs or ramps.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..palettes import PREFIX, custom_lut, load_palettes, palette_token, read_palette


def builtin_palettes() -> tuple[tuple[str, str], ...]:
    """
    List stable software palette identifiers and their current application labels.
    """
    from ..pipeline_model.catalog import PALETTES

    result = tuple(PALETTES.items())
    return result


def palette_key(value: Any) -> str:
    """
    Validate a built-in identifier, legacy visualization label or portable custom snapshot.
    """
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - use uniform schema errors for declarative controls
            "Palette must be a software palette identifier or custom gradient"
        )
    if value.startswith(PREFIX):
        read_palette(value)
        return value
    builtins = dict(builtin_palettes())
    if value in builtins:
        return value
    aliases = {f"{label} (approx.)" if key.startswith("camera_") else label: key
               for key, label in builtins.items()}
    aliases["Grayscale"] = "white_hot"
    if value not in aliases:
        raise ValueError(f"Unknown software palette: {value!r}")
    return aliases[value]


def palette_choices() -> tuple[tuple[str, str], ...]:
    """
    Discover built-ins and saved custom palettes on demand, without importing package code.

    Values can be saved directly in a palette control. Custom values embed their full
    gradient, so exported pipelines work independently of the recipient's library.
    Call for discovery, rather than on every processed frame.
    """
    custom = tuple((palette_token(value), f"Custom · {name}")
                   for name, value in sorted(load_palettes().items(), key=lambda pair: pair[0].casefold()))
    result = builtin_palettes() + custom
    return result


@lru_cache(maxsize=32)
def _builtin_lut(key: str) -> np.ndarray:
    """
    Sample the existing software renderer once into an immutable BGR lookup table.
    """
    from ..processing.images import colorize

    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    result = colorize(ramp, key)[:, 0].copy()
    result.setflags(write=False)
    return result


def palette_lut(value: str) -> np.ndarray:
    """
    Return a cached immutable 256×3 uint8 BGR table without reading the user library.
    """
    key = palette_key(value)
    result = custom_lut(key) if key.startswith(PREFIX) else _builtin_lut(key)
    return result


def apply_palette(gray: np.ndarray, value: str) -> np.ndarray:
    """
    Colorize a finite two-dimensional 0–255 display-intensity plane, leaving it untouched.

    Output is uint8 BGR. Values outside the range are clipped and fractional values
    rounded, matching the software color node. No temperature conversion is performed.
    """
    if (not isinstance(gray, np.ndarray) or gray.ndim != 2 or not gray.size
            or not np.issubdtype(gray.dtype, np.number) or np.iscomplexobj(gray)
            or not np.isfinite(gray).all()):
        raise ValueError("Palette input must be a nonempty finite numeric 2D display plane")
    indices = np.clip(gray, 0, 255).round().astype(np.uint8)
    result = palette_lut(value)[indices]
    return result
