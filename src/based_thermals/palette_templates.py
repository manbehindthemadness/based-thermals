"""
Editable copies of the same built-in software gradients used by pipeline nodes.
"""

from typing import Any

import numpy as np

from .palettes import MAX_STOPS, validate_palette


def builtin_template(key: str) -> dict[str, Any]:
    """
    Sample the node renderer and retain enough stops to reproduce its complete LUT exactly.

    Built-in definitions remain unchanged; editing and saving creates a custom palette.
    """
    from .pipeline_model.catalog import PALETTES
    from .processing.images import colorize

    if key not in PALETTES:
        raise ValueError("Choose a built-in software palette")
    positions = np.linspace(0, 1, 256)
    ramp = colorize(np.arange(256, dtype=np.uint8).reshape(256, 1), key)[:, 0, ::-1]
    selected = [0, 255]
    while len(selected) < MAX_STOPS:
        interpolated = np.stack([
            np.interp(positions, positions[selected], ramp[selected, channel])
            for channel in range(3)
        ], axis=1).round()
        error = np.max(np.abs(interpolated - ramp), axis=1)
        index = int(np.argmax(error))
        if error[index] == 0:
            break
        selected.append(index)
        selected.sort()
    stops = [{"position": float(positions[index]),
              "color": "#" + "".join(f"{channel:02x}" for channel in ramp[index])}
             for index in selected]
    result = validate_palette({"name": f"{PALETTES[key]} copy", "stops": stops})
    return result
