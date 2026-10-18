"""
Display-only annotation selection and source-local control geometry.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MarkupSelection:
    """
    Retain a whole-object or control-point selection after mouse release.
    """

    node: str
    identity: str
    index: int = -1
    matrix: tuple[float, ...] = ()


def handles(item: dict[str, Any]) -> list[list[float]]:
    """
    Expose polygon vertices, endpoints, label anchors or all four box/ellipse bounds corners.
    """
    points = item["points"]
    if not points:
        return []
    if item["kind"] in ("rectangle", "textbox", "ellipse", "table"):
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        return [low.tolist(), [float(high[0]), float(low[1])], high.tolist(),
                [float(low[0]), float(high[1])]]
    return [list(point) for point in points]


def move_handle(item: dict[str, Any], index: int, point: list[float]) -> list[list[float]]:
    """
    Move one control inside the source canvas while retaining the opposite bounding corner.
    """
    controls = handles(item)
    if not 0 <= index < len(controls):
        raise ValueError("Markup control point no longer exists")
    moved = np.clip(point, 0, 1).tolist()
    if item["kind"] in ("rectangle", "textbox", "ellipse", "table"):
        return [moved, controls[(index + 2) % 4]]
    controls[index] = moved
    return controls
