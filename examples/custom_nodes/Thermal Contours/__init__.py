# ruff: noqa: N999
"""
Render a display-only thermal visualization in an isolated Custom node.
"""

from typing import Any

import numpy as np

from .contours import render

CONFIG_JSON = {
    "defaults": {
        "palette": "turbo",
        "normalization": "Source range",
        "smoothing": 1,
        "levels": 12,
        "bands": True,
        "lines": True,
        "line_width": 1,
        "line_color": "dynamic",
        "line_opacity": 0.7,
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
            "key": "levels",
            "label": "Contour intervals",
            "type": "integer",
            "min": 2,
            "max": 64,
            "step": 1,
        },
        {"key": "bands", "label": "Filled color bands", "type": "boolean"},
        {"key": "lines", "label": "Show contour lines", "type": "boolean"},
        {
            "key": "line_width",
            "label": "Line width (px)",
            "type": "integer",
            "min": 1,
            "max": 5,
            "step": 1,
        },
        {"key": "line_color", "label": "Contour color", "type": "color"},
        {
            "key": "line_opacity",
            "label": "Contour opacity",
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
