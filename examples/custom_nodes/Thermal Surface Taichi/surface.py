# Intentional baseline geometry/shading/fallback shared by the controlled renderer comparison.
# noinspection DuplicatedCode
"""
Render a bounded thermal height mesh using vectorized geometry and OpenCV rasterization.
"""

from typing import Any, cast

import numpy as np

from based_thermals.custom_nodes.palettes import palette_lut

from .antialiasing import rasterize
from .geometry import ByteArray, Geometry
from .mesh import heat_values, lit_triangles
from .raster import LOG, Rasterizer
from .settings import Settings


# noinspection DuplicatedCode
def palette(name: str) -> ByteArray:
    """
    Read the app's cached BGR ramp without loading assets or initializing a device.
    """
    colors = palette_lut(name)
    return colors


# noinspection DuplicatedCode
class SurfaceRenderer:
    """
    Retain one mesh/view and color ramp, without frame history or independent workers.
    """

    def __init__(self) -> None:
        """
        Initialize private bounded caches for this Custom node instance.
        """
        self.geometry: Geometry | None = None
        self.geometry_key: tuple[Any, ...] | None = None
        self.palette_name = ""
        self.colors = cast(ByteArray, np.empty((0, 3), dtype=np.uint8))
        self.rasterizer: Rasterizer | None = None
        self.execution = ""
        self.geometry_method = ""
        self.effective_execution = "OpenCV"
        self.fallback_reason = ""

    def render(self, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """
        Replace the input with a lit heat surface, preserving the caller's image.
        """
        settings = Settings.read(config)
        execution = settings.execution
        if self.execution != execution or self.geometry_method != settings.geometry_method:
            self.rasterizer = None
            self.execution = execution
            self.geometry_method = settings.geometry_method
            self.fallback_reason = ""
            self.effective_execution = "OpenCV"
        image_height, image_width = image.shape[:2]
        shape = image_height, image_width
        key = (shape, settings.detail, settings.yaw, settings.tilt,
               settings.height, settings.resolution, settings.scene_rotation,
               settings.offset_x, settings.offset_y, settings.zoom)
        geometry = self.geometry
        if geometry is None or self.geometry_key != key:
            geometry = Geometry(shape, settings)
            self.geometry = geometry
            self.geometry_key = key
        if self.palette_name != settings.palette:
            self.colors = palette(settings.palette)
            self.palette_name = settings.palette
        heat = heat_values(image, geometry, settings)
        sample_scale = 1 if settings.antialiasing == "Off" else 2
        if execution != "OpenCV" and not self.fallback_reason:
            try:
                rasterizer = self.rasterizer
                if rasterizer is None:
                    rasterizer = Rasterizer(execution)
                    self.rasterizer = rasterizer
                if settings.geometry_method == "Taichi":
                    result = rasterizer.render_mesh(heat, geometry, settings, self.colors, sample_scale)
                else:
                    triangles, colors = lit_triangles(heat, geometry, settings, self.colors)
                    result = rasterizer.render(geometry.floor, triangles, colors, settings.wireframe,
                                               sample_scale)
                self.effective_execution = execution
                return result
            except Exception as exc:  # noqa: BLE001 - Optional compilers have their own exception types.
                # Optional native backends can raise compiler-specific exceptions; retain CPU output.
                self.rasterizer = None
                self.effective_execution = "OpenCV"
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                LOG.warning("Thermal Surface using OpenCV: %s", self.fallback_reason)
        triangles, colors = lit_triangles(heat, geometry, settings, self.colors)
        result = rasterize(geometry.floor, triangles, colors, settings.wireframe, sample_scale)
        return result
