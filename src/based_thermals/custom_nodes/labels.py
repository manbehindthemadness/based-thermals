"""
Collect custom-node labels for upright rendering at the final viewer resolution.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..desktop_app.typography import LABEL_FONT_SCALE


@dataclass(frozen=True)
class DisplayLabel:
    """
    Store text and its normalized anchor independently of image resolution.
    """

    text: str
    x: float
    y: float

    def transformed(self, matrix: np.ndarray) -> DisplayLabel:
        """
        Move an anchor while allowing richer deferred annotations to retain their geometry.
        """
        anchor = matrix @ np.array((self.x, self.y, 1))
        return replace(self, x=float(anchor[0] / anchor[2]), y=float(anchor[1] / anchor[2]))


_collector: ContextVar[list[DisplayLabel] | None] = ContextVar("custom_labels", default=None)


@contextmanager
def collect_labels(labels: list[DisplayLabel]) -> Iterator[None]:
    """
    Isolate each branch worker's annotation collector, including failure cleanup.
    """
    token = _collector.set(labels)
    try:
        yield
    finally:
        _collector.reset(token)


def defer_labels(labels: list[tuple[str, tuple[int, int]]], shape: tuple[int, int]) -> bool:
    """
    Queue pixel anchors in a Custom worker; return False for standalone callers.
    """
    collector = _collector.get()
    if collector is None:
        return False
    height, width = shape
    for text, (x, y) in labels:
        if not isinstance(text, str) or len(text) > 256 or not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Display labels require bounded text and an in-image pixel anchor")
        collector.append(DisplayLabel(text, x / max(1, width - 1), y / max(1, height - 1)))
    return True


def transform_labels(
    labels: list[DisplayLabel], rotation: int = 0, horizontal: bool = False, vertical: bool = False,
) -> list[DisplayLabel]:
    """
    Move anchors with image geometry while leaving glyphs upright and unmirrored.
    """
    if rotation % 360 == 0 and not horizontal and not vertical:
        return list(labels)
    transformed = []
    matrix = np.eye(3)
    for _ in range(rotation // 90):
        matrix = np.array(((0, -1, 1), (1, 0, 0), (0, 0, 1))) @ matrix
    mirror = np.array(((-1 if horizontal else 1, 0, int(horizontal)),
                       (0, -1 if vertical else 1, int(vertical)), (0, 0, 1)))
    for label in labels:
        transformed.append(label.transformed(mirror @ matrix))
    return transformed


def confidence_mask(
    shape: tuple[int, int], labels: list[tuple[str, tuple[int, int]]],
) -> np.ndarray:
    """
    Use measurement typography and layout, shrinking only for very small images.
    """
    from ..desktop_app.overlays import place_temperature_label

    height, width = shape
    mask = np.zeros(shape, np.uint8)
    occupied: list[tuple[int, int, int, int]] = []
    for text, anchor in labels:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, 1
        )
        if width >= text_width + 6 and height >= text_height + baseline + 6:
            origin, bounds = place_temperature_label(text, anchor, (height, width, 3), occupied)
            occupied.append(bounds)
            scale = LABEL_FONT_SCALE
        else:
            scale = LABEL_FONT_SCALE * min(max(1, width - 4) / max(1, text_width),
                                           max(1, height - 4) / max(1, text_height + baseline))
            (_, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            origin = (min(2, width - 1), min(height - 1, text_height + 1))
        cv2.putText(mask, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    return mask


def draw_display_labels(
    image: np.ndarray, labels: list[DisplayLabel], rotation: int, *, markup: bool = True,
) -> np.ndarray:
    """
    Composite labels after all pipeline resizing and viewer rotation.
    """
    from ..desktop_app.overlays import draw_contrasting_overlay
    from ..markup.labels import MarkupLabel
    from ..markup.paint import draw_labels

    if not labels:
        return image
    height, width = image.shape[:2]
    visible = labels if markup else [label for label in labels if not isinstance(label, MarkupLabel)]
    transformed = transform_labels(visible, rotation)
    anchors = [(label.text, (round(label.x * (width - 1)), round(label.y * (height - 1))))
               for label in transformed if not isinstance(label, MarkupLabel)]
    output = image
    if anchors:
        text = confidence_mask((height, width), anchors)
        output = draw_contrasting_overlay(image, np.zeros_like(text), text)
    if markup:
        output = draw_labels(output, transformed)
    return output
