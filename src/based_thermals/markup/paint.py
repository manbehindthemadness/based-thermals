"""
Bounded ROI mask caching and measurement-style color compositing for pipeline markup.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

import cv2
import numpy as np

from ..custom_nodes.labels import DisplayLabel
from .composite import blend, contrast
from .labels import MarkupLabel
from .tables import paint_table


@dataclass(frozen=True)
class Masks:
    """
    Retain only the clipped annotation footprint, never a history of full-size frame masks.
    """

    x: int
    y: int
    border: np.ndarray
    fill: np.ndarray
    text: np.ndarray


_CACHE: OrderedDict[tuple, Masks] = OrderedDict()
_LOCK = RLock()
_BYTES = 0
MAX_BYTES = 64 * 1024 * 1024


def project(points: list, matrix: tuple[float, ...]) -> np.ndarray:
    """
    Map normalized source points through their branch and viewer transforms.
    """
    values = np.asarray(points, np.float64)
    moved = np.column_stack((values, np.ones(len(values)))) @ np.asarray(matrix).reshape(3, 3).T
    return moved[:, :2] / moved[:, 2:]


def masks_for(shape: tuple[int, int], item: dict[str, Any], matrix: tuple[float, ...]) -> Masks:
    """
    Reuse masks across changing camera frames, colors and opacities, with a shared memory cap.
    """
    global _BYTES
    key = (shape, item["kind"], tuple(tuple(p) for p in item["points"]), item["text"],
           item["font_size"], item["width"], matrix)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    result = rasterize(shape, item, matrix)
    size = result.border.nbytes + result.fill.nbytes + result.text.nbytes
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        while _CACHE and _BYTES + size > MAX_BYTES:
            _, old = _CACHE.popitem(last=False)
            _BYTES -= old.border.nbytes + old.fill.nbytes + old.text.nbytes
        if size <= MAX_BYTES:
            _CACHE[key] = result
            _BYTES += size
        while len(_CACHE) > 256:
            _, old = _CACHE.popitem(last=False)
            _BYTES -= old.border.nbytes + old.fill.nbytes + old.text.nbytes
    return result


def rasterize(shape: tuple[int, int], item: dict[str, Any], matrix: tuple[float, ...]) -> Masks:
    """
    Rasterize bounded display geometry and upright multiline text into a local mask rectangle.
    """
    height, width = shape
    kind = item["kind"]
    points = np.asarray(item["points"], np.float64)
    if kind in ("rectangle", "textbox", "ellipse", "table"):
        low, high = points.min(axis=0), points.max(axis=0)
        if kind == "ellipse":
            angles = np.linspace(0, 2 * np.pi, 65)[:-1]
            points = (low + high) / 2 + np.column_stack((np.cos(angles), np.sin(angles))) * (high - low) / 2
        else:
            points = np.array((low, (high[0], low[1]), high, (low[0], high[1])))
    mapped = project(points.tolist(), matrix) * (max(1, width - 1), max(1, height - 1))
    mapped = np.clip(np.rint(mapped), -65536, 65536).astype(np.int32)
    left, top = mapped.min(axis=0)
    right, bottom = mapped.max(axis=0)
    scale = item["font_size"] / 22
    lines = item["text"].splitlines() or [""]
    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines)
    line_height = max(1, round(item["font_size"] * 1.4))
    if kind == "text":
        right, bottom = left + text_width + 12, top + line_height * len(lines) + 8
        mapped = np.array(((left, top), (right, top), (right, bottom), (left, bottom)), np.int32)
    padding = max(18, item["width"] + 4)
    x0, y0 = max(0, int(left) - padding), max(0, int(top) - padding)
    x1, y1 = min(width, int(right) + padding + 1), min(height, int(bottom) + padding + 1)
    size = max(0, y1 - y0), max(0, x1 - x0)
    border, filled, text = (np.zeros(size, np.uint8) for _ in range(3))
    if not border.size:
        return Masks(x0, y0, border, filled, text)
    local = mapped - (x0, y0)
    if kind in ("line", "arrow"):
        start, end = tuple(local[0]), tuple(local[1])
        if kind == "arrow":
            length = max(1., float(np.linalg.norm(local[1] - local[0])))
            cv2.arrowedLine(border, start, end, 255, item["width"], cv2.LINE_AA,
                            tipLength=min(.45, 18 / length))
        else:
            cv2.line(border, start, end, 255, item["width"], cv2.LINE_AA)
    else:
        cv2.fillPoly(filled, [local], 255, cv2.LINE_AA)
        cv2.polylines(border, [local], True, 255, item["width"], cv2.LINE_AA)
    if kind == "table":
        paint_table(border, text, item, matrix, shape, (x0, y0))
    elif item["text"]:
        origin = (int(left) - x0 + 6, int(top) - y0 + item["font_size"] + 4)
        for index, line in enumerate(lines):
            cv2.putText(text, line, (origin[0], origin[1] + index * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
        if kind != "text":
            cv2.bitwise_and(text, filled, dst=text)
    for mask in (border, filled, text):
        mask.setflags(write=False)
    return Masks(x0, y0, border, filled, text)


def draw_object(image: np.ndarray, item: dict[str, Any], matrix: tuple[float, ...]) -> None:
    """
    Paint one cached object in-place using independent style and alpha channels.
    """
    masks = masks_for(image.shape[:2], item, matrix)
    if not masks.border.size:
        return
    view = image[masks.y:masks.y + masks.border.shape[0], masks.x:masks.x + masks.border.shape[1]]
    if item["fill"]:
        blend(view, masks.fill, item["fill_color"], item["fill_opacity"])
    for role, mask in (("border", masks.border), ("text", masks.text)):
        alpha = item[f"{role}_opacity"]
        if alpha == 0 or not np.any(mask):
            continue
        color = item[f"{role}_color"]
        if color == "dynamic":
            contrast(view, mask, alpha, text=role == "text")
        else:
            blend(view, mask, color, alpha)


def draw_labels(image: np.ndarray, labels: list[DisplayLabel]) -> np.ndarray:
    """
    Composite placed annotation labels independently of all temperature data and pixel filters.
    """
    objects = [label for label in labels if isinstance(label, MarkupLabel) and label.definition]
    objects.sort(key=lambda label: label.definition["z_index"]
                 if isinstance(label, MarkupLabel) and label.definition is not None else 0)
    if not objects:
        return image
    result = image.copy()
    for label in objects:
        if isinstance(label, MarkupLabel) and label.definition is not None:
            draw_object(result, label.definition, label.matrix)
    return result


def hit_test(shape: tuple[int, int], item: dict[str, Any], matrix: tuple[float, ...],
             point: tuple[int, int]) -> bool:
    """
    Hit visible text/lines or a closed shape's interior without querying measurements.
    """
    masks = masks_for(shape, item, matrix)
    x, y = point[0] - masks.x, point[1] - masks.y
    if not 0 <= y < masks.border.shape[0] or not 0 <= x < masks.border.shape[1]:
        return False
    region = masks.border[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
    return bool(masks.fill[y, x] or masks.text[y, x] or np.any(region))
