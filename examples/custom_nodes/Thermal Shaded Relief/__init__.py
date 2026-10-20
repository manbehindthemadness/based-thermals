# ruff: noqa: N999
"""
Render a display-only thermal visualization in an isolated Custom node.
"""

from typing import Any

import numpy as np

from .relief import render

CONFIG_JSON = {
    "defaults": {
        "palette": "Turbo",
        "normalization": "Source range",
        "smoothing": 1,
        "azimuth": 315,
        "elevation": 45,
        "strength": 6,
        "ambient": 0.3,
        "mix": 1,
    },
    "controls": [
        {"key": "palette", "label": "Thermal palette", "type": "palette"},
        {
            "key": "normalization",
            "label": "Display range",
            "type": "choice",
            "options": ["Source range", "Auto contrast"],
        },
        {
            "key": "smoothing",
            "label": "Heat smoothing",
            "type": "number",
            "min": 0,
            "max": 5,
            "step": 0.1,
            "decimals": 1,
        },
        {
            "key": "azimuth",
            "label": "Light azimuth",
            "type": "number",
            "min": 0,
            "max": 360,
            "step": 5,
            "decimals": 1,
        },
        {
            "key": "elevation",
            "label": "Light elevation",
            "type": "number",
            "min": 5,
            "max": 90,
            "step": 5,
            "decimals": 1,
        },
        {
            "key": "strength",
            "label": "Relief exaggeration",
            "type": "number",
            "min": 0,
            "max": 50,
            "step": 0.5,
            "decimals": 1,
        },
        {
            "key": "ambient",
            "label": "Ambient light",
            "type": "number",
            "min": 0,
            "max": 1,
            "step": 0.1,
            "decimals": 1,
        },
        {
            "key": "mix",
            "label": "Relief mix",
            "type": "number",
            "min": 0,
            "max": 1,
            "step": 0.1,
            "decimals": 1,
        },
    ],
}


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Transform display intensities without accessing camera measurements.
    """
    result = render(image, config)
    return result
