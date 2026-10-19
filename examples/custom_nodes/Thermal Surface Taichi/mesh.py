"""
Keep the original heat sampling and face lighting independent of the experimental rasterizer.
"""

import cv2
import numpy as np

from .geometry import ByteArray, FloatArray, Geometry, IndexArray
from .settings import Settings


# Portable baseline arithmetic intentionally matches the sibling Thermal Surface.
# noinspection DuplicatedCode
def heat_values(image: np.ndarray, geometry: Geometry, settings: Settings) -> FloatArray:
    """
    Sample grayscale display brightness using the original normalization and smoothing order.
    """
    sampled = cv2.resize(image, (geometry.columns, geometry.rows), interpolation=cv2.INTER_AREA)
    small: FloatArray = np.asarray(sampled, np.float32)
    if np.max(np.ptp(small, axis=2)) > 1:
        raise ValueError("Thermal Surface needs grayscale input: select Source Raw before this node")
    heat: FloatArray = small[:, :, 0]
    if settings.smoothing:
        heat = cv2.GaussianBlur(heat, (0, 0), settings.smoothing, borderType=cv2.BORDER_REPLICATE)
    if settings.normalization == "Auto contrast":
        low, high = np.percentile(heat, (1, 99))
        heat = (heat - low) / max(float(high - low), 1.0)
    else:
        heat = heat / 255.0
    result = np.clip(heat, 0, 1).ravel()
    return result


# noinspection DuplicatedCode
def lit_triangles(heat: FloatArray, geometry: Geometry, settings: Settings,
                  palette: ByteArray) -> tuple[IndexArray | FloatArray, ByteArray]:
    """
    Return painter-ordered screen triangles and BGR colors with unchanged face lighting.
    """
    count = len(heat)
    heights: FloatArray = np.concatenate((heat * settings.height, np.zeros(count, np.float32)))
    vertices: FloatArray = np.column_stack((np.tile(geometry.xy, (2, 1)), heights))
    projected, depth = geometry.project(vertices)
    faces: FloatArray = vertices[geometry.triangles]
    normals: FloatArray = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths: FloatArray = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-8)
    light: FloatArray = np.array((-0.35, -0.45, 1.0), np.float32)
    light /= np.linalg.norm(light)
    shade: FloatArray = 0.38 + 0.62 * np.clip(normals @ light, 0, 1)
    levels: FloatArray = np.concatenate((heat, heat))
    indices: ByteArray = np.rint(levels[geometry.triangles].mean(axis=1) * 255).astype(np.uint8)
    colors: ByteArray = (palette[indices] * shade[:, None]).astype(np.uint8)
    order: IndexArray = np.argsort(-depth[geometry.triangles].mean(axis=1), kind="stable")
    coordinates = projected[geometry.triangles][order]
    triangles: IndexArray | FloatArray
    if settings.antialiasing == "Off":
        triangles = np.rint(coordinates).astype(np.int32)
    else:
        triangles = coordinates.astype(np.float32)
    result = triangles, colors[order]
    return result
