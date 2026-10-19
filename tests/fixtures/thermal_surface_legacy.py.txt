# This folder names a Custom node; runtime generates its Python module name.
# ruff: noqa: N999
"""
Replace a grayscale thermal image with a shaded three-dimensional height surface.
"""

from typing import Any

import numpy as np

from .surface import SurfaceRenderer

CONFIG_JSON = {
    "defaults": {
        "detail": 40, "yaw": -35, "tilt": 50, "height": 0.7,
        "smoothing": 0.8, "palette": "Inferno", "normalization": "Source range",
        "resolution": "640", "wireframe": False,
        "scene_rotation": 0, "offset_x": 0, "offset_y": 0, "zoom": 1,
    },
    "controls": [
        {"key": "detail", "label": "Mesh detail", "type": "integer",
         "min": 12, "max": 96, "step": 4},
        {"key": "yaw", "label": "View rotation", "type": "number",
         "min": -180, "max": 180, "step": 5},
        {"key": "tilt", "label": "View elevation", "type": "number",
         "min": 15, "max": 85, "step": 5},
        {"key": "scene_rotation", "label": "Scene rotation", "type": "number",
         "min": -180, "max": 180, "step": 5},
        {"key": "offset_x", "label": "X offset (%)", "type": "number",
         "min": -100, "max": 100, "step": 1},
        {"key": "offset_y", "label": "Y offset (%)", "type": "number",
         "min": -100, "max": 100, "step": 1},
        {"key": "zoom", "label": "Zoom", "type": "number",
         "min": 0.1, "max": 5, "step": 0.1},
        {"key": "height", "label": "Height exaggeration", "type": "number",
         "min": 0, "max": 2, "step": 0.05},
        {"key": "smoothing", "label": "Surface smoothing", "type": "number",
         "min": 0, "max": 3, "step": 0.1},
        {"key": "palette", "label": "Thermal palette", "type": "choice",
         "options": ["Inferno", "Turbo", "Grayscale"]},
        {"key": "normalization", "label": "Height range", "type": "choice",
         "options": ["Source range", "Auto contrast"]},
        {"key": "resolution", "label": "Output long edge", "type": "choice",
         "options": ["384", "640", "960"]},
        {"key": "wireframe", "label": "Show mesh edges", "type": "boolean"},
    ],
}

_renderer = SurfaceRenderer()


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Render relative input brightness as elevation, without accessing native radiometry.
    """
    result = _renderer.render(image, config)
    return result
