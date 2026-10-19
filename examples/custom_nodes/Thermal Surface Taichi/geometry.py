# Portable baseline snapshot: embedded Custom packages cannot import sibling example folders.
# noinspection DuplicatedCode
"""
Cache an orthographic camera, triangle topology and the static floor per node.
"""

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .settings import Settings

FloatArray = npt.NDArray[np.floating[Any]]
IndexArray = npt.NDArray[np.integer[Any]]
ByteArray = npt.NDArray[np.uint8]


# noinspection DuplicatedCode
def topology(rows: int, columns: int) -> IndexArray:
    """
    Preserve the original top-face and clockwise side-wall triangle order.
    """
    indices = np.arange(rows * columns).reshape(rows, columns)
    a, b = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel()
    c, d = indices[1:, :-1].ravel(), indices[1:, 1:].ravel()
    top = np.concatenate((np.column_stack((a, d, b)), np.column_stack((a, c, d))))
    edges = (indices[0], indices[:, -1], indices[-1, ::-1], indices[::-1, 0])
    sides = []
    count = rows * columns
    for edge in edges:
        start, end = edge[:-1], edge[1:]
        sides.extend((np.column_stack((start, end, end + count)),
                      np.column_stack((start, end + count, start + count))))
    result = np.concatenate((top, *sides))
    return result


# noinspection DuplicatedCode
class Geometry:
    """
    Bound mesh size independently of input resolution and retain only the current view.
    """

    def __init__(self, shape: tuple[int, int], settings: Settings) -> None:
        """
        Map image rows into an upright world plane and build its fitted view and mesh.
        """
        height, width = shape
        aspect = np.array((width, height), np.float32) / max(width, height)
        grid_size = np.maximum(2, np.rint(settings.detail * aspect)).astype(int)
        self.columns, self.rows = int(grid_size[0]), int(grid_size[1])
        x, y = np.meshgrid(
            np.linspace(-aspect[0], aspect[0], self.columns, dtype=np.float32),
            np.linspace(aspect[1], -aspect[1], self.rows, dtype=np.float32),
        )
        self.xy: FloatArray = np.column_stack((x.ravel(), y.ravel()))
        yaw, tilt = np.deg2rad((settings.yaw, settings.tilt))
        self.rotation: FloatArray = np.array(((np.cos(yaw), -np.sin(yaw)),
                                  (np.sin(yaw), np.cos(yaw))), np.float32)
        roll = np.deg2rad(settings.scene_rotation)
        self.screen_rotation: FloatArray = np.array(((np.cos(roll), -np.sin(roll)),
                                         (np.sin(roll), np.cos(roll))), np.float32)
        self.sin_tilt, self.cos_tilt = float(np.sin(tilt)), float(np.cos(tilt))
        output_size = np.maximum(
            32, np.rint(settings.resolution * aspect)
        ).astype(int)
        pixel_count = int(np.prod(output_size))
        if pixel_count > 4_000_000:
            output_size = np.floor(output_size * np.sqrt(4_000_000 / pixel_count)).astype(int)
        self.output_width, self.output_height = int(output_size[0]), int(output_size[1])
        self.scale = 1.0
        self.offset: FloatArray = np.zeros(2, np.float32)
        self.fit(aspect, settings)
        self.triangles = topology(self.rows, self.columns)
        self.floor = self.make_floor(aspect)

    def fit(self, aspect: FloatArray, settings: Settings) -> None:
        """
        Fit the same bounding prism before applying user zoom and pan.
        """
        corners: FloatArray = np.array([
            (x, y, z) for x in (-aspect[0], aspect[0])
            for y in (-aspect[1], aspect[1]) for z in (0, settings.height)
        ], np.float32)
        projected, _ = self.project(corners)
        low, high = projected.min(axis=0), projected.max(axis=0)
        available = np.array((self.output_width, self.output_height)) * 0.82
        self.scale = float(np.min(available / np.maximum(high - low, 1e-6))) * settings.zoom
        self.offset = (
            np.array((self.output_width, self.output_height), np.float32) / 2
            - (low + high) * (self.scale / 2)
            + np.array((self.output_width * settings.offset_x,
                        self.output_height * settings.offset_y), np.float32) / 100
        )

    def project(self, vertices: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Apply camera roll, zoom and screen-space pan without changing surface depth.
        """
        rotated: FloatArray = vertices[:, :2] @ self.rotation.T
        z: FloatArray = vertices[:, 2]
        pixels: FloatArray = np.column_stack((
            rotated[:, 0], -rotated[:, 1] * self.sin_tilt - z * self.cos_tilt,
        ))
        pixels = pixels @ self.screen_rotation.T
        depth: FloatArray = rotated[:, 1] * self.cos_tilt - z * self.sin_tilt
        return pixels * self.scale + self.offset, depth

    def make_floor(self, aspect: FloatArray) -> ByteArray:
        """
        Cache a subdued ground grid under the changing surface.
        """
        canvas = np.full((self.output_height, self.output_width, 3), (18, 15, 12), np.uint8)
        aspect = aspect * 1.06
        lines = []
        for t in np.linspace(-1, 1, 9):
            vertices = np.array(((-aspect[0], t * aspect[1], 0),
                                 (aspect[0], t * aspect[1], 0),
                                 (t * aspect[0], -aspect[1], 0),
                                 (t * aspect[0], aspect[1], 0)), np.float32)
            projected, _ = self.project(vertices)
            lines.extend(np.rint(projected).astype(np.int32).reshape(2, 2, 2))
        cv2.polylines(canvas, lines, False, (43, 39, 33), 1, cv2.LINE_AA)
        return canvas
