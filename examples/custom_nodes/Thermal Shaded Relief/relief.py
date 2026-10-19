"""
Apply a top-down hillshade to relative heat while preserving sensor alignment.
"""

from typing import Any

import cv2
import numpy as np

from based_thermals.custom_nodes.visualization import (
    display_settings,
    grayscale,
    normalized,
    number,
)


def render(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Light float32 Sobel normals with a configurable distant directional light.
    """
    colors, mode, smoothing = display_settings(config)
    azimuth = np.deg2rad(number(config, "azimuth", 315, 0, 360))
    elevation = np.deg2rad(number(config, "elevation", 45, 5, 90))
    strength = number(config, "strength", 6, 0, 50)
    ambient = number(config, "ambient", 0.3, 0, 1)
    mix = number(config, "mix", 1, 0, 1)
    heat = normalized(grayscale(image, smoothing), mode)
    scale = strength * min(heat.shape) / 8
    dx = cv2.Sobel(heat, cv2.CV_32F, 1, 0, ksize=3, scale=scale, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(heat, cv2.CV_32F, 0, 1, ksize=3, scale=scale, borderType=cv2.BORDER_REPLICATE)
    light_x = np.cos(azimuth) * np.cos(elevation)
    light_y = np.sin(azimuth) * np.cos(elevation)
    incidence = (-dx * light_x - dy * light_y + np.sin(elevation)) / np.sqrt(1 + dx * dx + dy * dy)
    shade = ambient + (1 - ambient) * np.clip(incidence, 0, 1)
    gain = 1 - mix + mix * shade
    canvas = colors[np.rint(heat * 255).astype(np.uint8)].astype(np.float32)
    result = canvas * gain[:, :, None]
    return result
