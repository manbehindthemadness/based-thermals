"""
Independent measurement-overlay references for fast sparse/dense annotation compositing.
"""

import cv2
import numpy as np
import pytest

from based_thermals.desktop_app.overlays import draw_contrasting_overlay
from based_thermals.markup.composite import blend, contrast
from based_thermals.measurements.appearance import blend_color


@pytest.mark.parametrize("opacity", [0., .2, .4, .5, .75, 1.])
@pytest.mark.parametrize("color", ["#a832ef", "#000000", "dynamic"])
@pytest.mark.parametrize("dense", [False, True])
def test_fast_blending_matches_measurement_alpha(opacity: float, color: str, dense: bool) -> None:
    """
    Preserve RGB/BGR conversion, antialiasing and rounding for both dense boxes and sparse strokes.
    """
    image = np.random.default_rng(37).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (3, 5), (117, 75), 255, -1 if dense else 1, cv2.LINE_AA)
    expected = blend_color(image, mask, color, opacity)
    result = image.copy()
    blend(result, mask, color, opacity)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("opacity", [.2, .5, 1.])
@pytest.mark.parametrize("text", [False, True])
def test_sparse_contrast_matches_existing_measurement_halos(opacity: float, text: bool) -> None:
    """
    Match exact inverted borders, middle-gray halos and outlined white text across busy images.
    """
    image = np.random.default_rng(52).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    image[:, 40:60] = 127
    mask = np.zeros(image.shape[:2], np.uint8)
    if text:
        cv2.putText(mask, "25.5 C", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(mask, (0, 0), (117, 75), 255, 1, cv2.LINE_AA)
    empty = np.zeros_like(mask)
    target = draw_contrasting_overlay(image, empty if text else mask, mask if text else empty)
    expected = cv2.addWeighted(image, 1 - opacity, target, opacity, 0)
    result = image.copy()
    contrast(result, mask, opacity, text=text)
    np.testing.assert_array_equal(result, expected)
