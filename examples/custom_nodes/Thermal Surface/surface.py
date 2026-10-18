"""
Render a bounded thermal height mesh using vectorized geometry and OpenCV rasterization.
"""

from typing import Any

import cv2
import numpy as np

from .geometry import Geometry
from .settings import Settings


def palette(name: str) -> np.ndarray:
    """
    Construct a BGR color ramp without loading assets or initializing a GPU runtime.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    if name == "Grayscale":
        return np.repeat(ramp, 3, axis=1)
    mode = cv2.COLORMAP_INFERNO if name == "Inferno" else cv2.COLORMAP_TURBO
    colors = cv2.applyColorMap(ramp, mode)[:, 0]
    return colors


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
        self.colors = np.empty((0, 3), np.uint8)

    def render(self, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """
        Replace the input with a lit heat surface, preserving the caller's image.
        """
        settings = Settings.read(config)
        shape = image.shape[:2]
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
        small = cv2.resize(image, (geometry.columns, geometry.rows), interpolation=cv2.INTER_AREA)
        small = np.asarray(small, np.float32)
        if np.max(np.ptp(small, axis=2)) > 1:
            raise ValueError(
                "Thermal Surface needs grayscale input: select Source Raw before this node"
            )
        heat = small[:, :, 0]
        if settings.smoothing:
            heat = cv2.GaussianBlur(
                heat, (0, 0), settings.smoothing, borderType=cv2.BORDER_REPLICATE,
            )
        if settings.normalization == "Auto contrast":
            low, high = np.percentile(heat, (1, 99))
            heat = (heat - low) / max(float(high - low), 1.0)
        else:
            heat = heat / 255.0
        heat = np.clip(heat, 0, 1).ravel()
        count = len(heat)
        heights = np.concatenate((heat * settings.height, np.zeros(count, np.float32)))
        vertices = np.column_stack((np.tile(geometry.xy, (2, 1)), heights))
        projected, depth = geometry.project(vertices)
        faces = vertices[geometry.triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-8)
        light = np.array((-0.35, -0.45, 1.0), np.float32)
        light /= np.linalg.norm(light)
        shade = 0.38 + 0.62 * np.clip(normals @ light, 0, 1)
        levels = np.concatenate((heat, heat))
        indices = np.rint(levels[geometry.triangles].mean(axis=1) * 255).astype(np.uint8)
        colors = (self.colors[indices] * shade[:, None]).astype(np.uint8)
        order = np.argsort(-depth[geometry.triangles].mean(axis=1), kind="stable")
        triangles = np.rint(projected[geometry.triangles][order]).astype(np.int32)
        canvas = geometry.floor.copy()
        for triangle, color in zip(triangles, colors[order].tolist(), strict=True):
            cv2.fillConvexPoly(canvas, triangle, color, cv2.LINE_8)
            if settings.wireframe:
                cv2.polylines(canvas, [triangle], True, (35, 33, 30), 1, cv2.LINE_8)
        result = canvas
        return result
