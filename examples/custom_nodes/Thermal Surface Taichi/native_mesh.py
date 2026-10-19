"""
Retain one node's compiled geometry buffers while preserving a stable host depth sort.
"""

from typing import Any

import numpy as np

from .geometry import Geometry
from .settings import Settings


class NativeMesh:
    """
    Keep topology and intermediate mesh arrays resident in the selected Taichi runtime.
    """

    def __init__(self, ti: Any, geometry: Geometry, settings: Settings) -> None:
        """
        Allocate only the current geometry, replacing this cache when the view or mesh changes.
        """
        self.geometry = geometry
        count, faces = len(geometry.xy), len(geometry.triangles)
        self.xy = ti.ndarray(ti.f32, (count, 2))
        self.xy.from_numpy(np.ascontiguousarray(geometry.xy, np.float32))
        self.faces = ti.ndarray(ti.i32, (faces, 3))
        self.faces.from_numpy(np.ascontiguousarray(geometry.triangles, np.int32))
        self.heat = ti.ndarray(ti.f32, (count,))
        self.vertices = ti.ndarray(ti.f32, (2 * count, 3))
        self.pixels = ti.ndarray(ti.f32, (2 * count, 2))
        self.depths = ti.ndarray(ti.f32, (2 * count,))
        self.face_depths = ti.ndarray(ti.f32, (faces,))
        self.triangles = ti.ndarray(ti.f32, (faces, 3, 2))
        self.colors = ti.ndarray(ti.u8, (faces, 3))
        self.ordered_triangles = ti.ndarray(ti.f32, (faces, 3, 2))
        self.ordered_colors = ti.ndarray(ti.u8, (faces, 3))
        self.camera = np.concatenate((geometry.rotation.ravel(), geometry.screen_rotation.ravel(),
                                      [geometry.sin_tilt, geometry.cos_tilt, geometry.scale,
                                       *geometry.offset, settings.height])).astype(np.float32)

    def prepare(self, library: Any, heat: np.ndarray, palette: np.ndarray,
                quantize: bool) -> tuple[Any, Any]:
        """
        Upload heat, compute projection and lighting, then read back only depths for stable sorting.
        """
        self.heat.from_numpy(np.ascontiguousarray(heat, np.float32))
        library.project(self.xy, self.heat, self.camera, self.vertices, self.pixels, self.depths)
        library.shade(self.faces, self.vertices, self.pixels, self.depths, self.heat,
                      np.ascontiguousarray(palette, np.uint8), self.triangles, self.colors,
                      self.face_depths, int(quantize))
        depths = self.face_depths.to_numpy()
        order = np.argsort(-depths, kind="stable").astype(np.int32)
        library.reorder(order, self.triangles, self.colors,
                        self.ordered_triangles, self.ordered_colors)
        return self.ordered_triangles, self.ordered_colors
