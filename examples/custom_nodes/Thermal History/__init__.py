# ruff: noqa: N999
"""
Render a display-only thermal visualization in an isolated Custom node.
"""

from typing import Any

import numpy as np

from .history import HistoryRenderer

OUTPUT_KIND = "visualization"

CONFIG_JSON = {
    "defaults": {
        "palette": "turbo",
        "normalization": "Source range",
        "smoothing": 1,
        "profile": "Mean rows",
        "position": 50,
        "samples": 96,
        "history": 64,
        "every": 1,
        "freeze": False,
        "reset": 0,
        "yaw": -25,
        "tilt": 50,
        "height": 0.7,
        "scene_rotation": 0,
        "offset_x": 0,
        "offset_y": 0,
        "zoom": 1,
        "resolution": "1280",
        "wireframe": False,
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
            "key": "profile",
            "label": "Profile source",
            "type": "choice",
            "options": ["Mean rows", "Mean columns", "Horizontal line", "Vertical line"],
        },
        {
            "key": "position",
            "label": "Line position (%)",
            "type": "number",
            "min": 0,
            "max": 100,
            "step": 1,
            "decimals": 1,
        },
        {
            "key": "samples",
            "label": "Profile samples",
            "type": "integer",
            "min": 16,
            "max": 256,
            "step": 8,
        },
        {
            "key": "history",
            "label": "History samples",
            "type": "integer",
            "min": 8,
            "max": 256,
            "step": 8,
        },
        {
            "key": "every",
            "label": "Record every N processed frames",
            "type": "integer",
            "min": 1,
            "max": 60,
            "step": 1,
        },
        {"key": "freeze", "label": "Freeze history", "type": "boolean"},
        {
            "key": "reset",
            "label": "Clear history (change value)",
            "type": "integer",
            "min": 0,
            "max": 1000000,
            "step": 1,
        },
        {
            "key": "yaw",
            "label": "View rotation",
            "type": "number",
            "min": -180,
            "max": 180,
            "step": 5,
            "decimals": 1,
        },
        {
            "key": "tilt",
            "label": "View elevation",
            "type": "number",
            "min": 15,
            "max": 85,
            "step": 5,
            "decimals": 1,
        },
        {
            "key": "height",
            "label": "Height exaggeration",
            "type": "number",
            "min": 0,
            "max": 2,
            "step": 0.1,
            "decimals": 1,
        },
        {
            "key": "scene_rotation",
            "label": "Scene rotation",
            "type": "number",
            "min": -180,
            "max": 180,
            "step": 5,
            "decimals": 1,
        },
        {
            "key": "offset_x",
            "label": "X offset (%)",
            "type": "number",
            "min": -100,
            "max": 100,
            "step": 1,
            "decimals": 1,
        },
        {
            "key": "offset_y",
            "label": "Y offset (%)",
            "type": "number",
            "min": -100,
            "max": 100,
            "step": 1,
            "decimals": 1,
        },
        {
            "key": "zoom",
            "label": "Zoom",
            "type": "number",
            "min": 0.1,
            "max": 5,
            "step": 0.1,
            "decimals": 1,
        },
        {
            "key": "resolution",
            "label": "Output long edge",
            "type": "choice",
            "options": ["384", "640", "960", "1280", "1600", "1920"],
        },
        {"key": "wireframe", "label": "Show mesh edges", "type": "boolean"},
    ],
}

_renderer = HistoryRenderer()


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Transform display intensities without accessing camera measurements.
    """
    result = _renderer.render(image, config)
    return result
