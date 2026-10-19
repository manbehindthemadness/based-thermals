"""
Rasterize the OpenCV fallback with optional subpixel supersampling.
"""

import cv2
import numpy as np


def rasterize(floor: np.ndarray, triangles: np.ndarray, colors: np.ndarray,
              wireframe: bool, sample_scale: int) -> np.ndarray:
    """
    Average four coverage samples when enabled, retaining the original output size.
    """
    height, width = floor.shape[:2]
    if sample_scale == 1:
        canvas = floor.copy()
        coordinates = triangles
    else:
        canvas = cv2.resize(floor, (width * sample_scale, height * sample_scale),
                            interpolation=cv2.INTER_NEAREST)
        coordinates = np.rint(triangles * sample_scale + (sample_scale - 1) * .5).astype(np.int32)
    for triangle, color in zip(coordinates, colors.tolist(), strict=True):
        cv2.fillConvexPoly(canvas, triangle, color, cv2.LINE_8)
        if wireframe:
            cv2.polylines(canvas, [triangle], True, (35, 33, 30), sample_scale, cv2.LINE_8)
    if sample_scale != 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return canvas
