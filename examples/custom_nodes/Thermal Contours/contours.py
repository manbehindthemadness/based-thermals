"""
Draw relative-heat contour bands and boundaries without moving image coordinates.
"""

from typing import Any

import cv2
import numpy as np

from based_thermals.custom_nodes.colors import color_bgr, color_value
from based_thermals.custom_nodes.visualization import (
    display_settings,
    flag,
    grayscale,
    normalized,
    number,
)


def render(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Quantize relative brightness and mark transitions in one bounded vectorized pass.
    """
    colors, mode, smoothing = display_settings(config)
    levels = int(number(config, "levels", 12, 2, 64, integer=True))
    width = int(number(config, "line_width", 1, 1, 5, integer=True))
    opacity = number(config, "line_opacity", 0.7, 0, 1)
    bands = flag(config, "bands", True)
    lines = flag(config, "lines", True)
    color = color_value(config.get("line_color", "dynamic"))
    heat = normalized(grayscale(image, smoothing), mode)
    indices = np.minimum((heat * levels).astype(np.uint8), levels - 1)
    values = (indices.astype(np.float32) + 0.5) / levels if bands else heat
    canvas = colors[np.rint(values * 255).astype(np.uint8)].copy()
    if lines and opacity:
        edges = np.zeros(heat.shape, np.uint8)
        edges[:, 1:] |= indices[:, 1:] != indices[:, :-1]
        edges[1:] |= indices[1:] != indices[:-1]
        if width > 1:
            edges = cv2.dilate(edges, np.ones((width, width), np.uint8))
        if color == "dynamic":
            from based_thermals.desktop_app.overlays import draw_contrasting_overlay

            overlay = draw_contrasting_overlay(canvas, edges * 255, np.zeros_like(edges))
            canvas = cv2.addWeighted(canvas, 1 - opacity, overlay, opacity, 0)
        else:
            mask = edges != 0
            ink = np.asarray(color_bgr(color))
            canvas[mask] = np.rint(canvas[mask] * (1 - opacity) + ink * opacity).astype(np.uint8)
    return canvas
