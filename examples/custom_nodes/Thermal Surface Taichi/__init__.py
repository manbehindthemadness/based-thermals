# This folder names a Custom node; runtime generates its Python module name.
# ruff: noqa: N999
"""
Compare an optional Taichi rasterizer with the original thermal height surface.
"""

from typing import Any

import numpy as np

from .surface import SurfaceRenderer

OUTPUT_KIND = "visualization"

CONFIG_JSON = {
    "defaults": {
        "execution": "Taichi CPU",
        "geometry_method": "NumPy",
        "detail": 256, "yaw": 80, "tilt": 45, "height": 0.7,
        "smoothing": 3, "palette": "Turbo", "normalization": "Source range",
        "resolution": "1280", "wireframe": False,
        "antialiasing": "2x supersampling",
        "scene_rotation": 90, "offset_x": -14, "offset_y": 0, "zoom": 1,
    },
    "controls": [
        {"key": "execution", "label": "Experimental renderer", "type": "choice",
         "options": ["OpenCV", "Taichi CPU", "Taichi Metal", "Taichi Vulkan", "Taichi CUDA"]},
        {"key": "geometry_method", "label": "Geometry + lighting", "type": "choice",
         "options": ["NumPy", "Taichi"]},
        {"key": "detail", "label": "Mesh detail", "type": "integer",
         "min": 12, "max": 512, "step": 4},
        {"key": "yaw", "label": "View rotation", "type": "number",
         "min": -180, "max": 180, "step": 5, "decimals": 1},
        {"key": "tilt", "label": "View elevation", "type": "number",
         "min": 15, "max": 85, "step": 5, "decimals": 1},
        {"key": "scene_rotation", "label": "Scene rotation", "type": "number",
         "min": -180, "max": 180, "step": 5, "decimals": 1},
        {"key": "offset_x", "label": "X offset (%)", "type": "number",
         "min": -100, "max": 100, "step": 1, "decimals": 1},
        {"key": "offset_y", "label": "Y offset (%)", "type": "number",
         "min": -100, "max": 100, "step": 1, "decimals": 1},
        {"key": "zoom", "label": "Zoom", "type": "number",
         "min": 0.1, "max": 5, "step": 0.1, "decimals": 1},
        {"key": "height", "label": "Height exaggeration", "type": "number",
         "min": 0, "max": 2, "step": 0.1, "decimals": 1},
        {"key": "smoothing", "label": "Surface smoothing", "type": "number",
         "min": 0, "max": 3, "step": 0.1, "decimals": 1},
        {"key": "palette", "label": "Thermal palette", "type": "choice",
         "options": ["Inferno", "Turbo", "Grayscale"]},
        {"key": "normalization", "label": "Height range", "type": "choice",
         "options": ["Source range", "Auto contrast"]},
        {"key": "resolution", "label": "Output long edge", "type": "choice",
         "options": ["384", "640", "960", "1280", "1600", "1920", "2304", "2560",
                     "3840", "4096"]},
        {"key": "wireframe", "label": "Show mesh edges", "type": "boolean"},
        {"key": "antialiasing", "label": "Antialiasing", "type": "choice",
         "options": ["Off", "2x supersampling"]},
    ],
}

_renderer = SurfaceRenderer()


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Render relative input brightness as elevation, without accessing native radiometry.
    """
    result = _renderer.render(image, config)
    return result
