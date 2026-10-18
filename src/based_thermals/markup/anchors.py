"""
Preserve existing label layouts while giving upright text a persistent source-local center.
"""

from typing import Any

import numpy as np

from .paint import label_bounds, project, text_layout


def center_offset(shape: tuple[int, int], item: dict[str, Any],
                  matrix: tuple[float, ...]) -> list[float] | None:
    """
    Convert a legacy displayed label rectangle to a center offset without moving its saved point.
    """
    try:
        inverse = np.linalg.inv(np.asarray(matrix).reshape(3, 3))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    lines, width, line_height = text_layout(item)
    left, top, right, bottom = label_bounds(shape, item, matrix, (width + 12, line_height * len(lines) + 8))
    if right <= left or bottom <= top:
        return None
    center = np.array(((left + right) / 2, (top + bottom) / 2))
    center /= (max(1, shape[1] - 1), max(1, shape[0] - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        source = project([center.tolist()], tuple(float(value) for value in inverse.ravel()))[0]
    offset = source - item["points"][0]
    if not np.isfinite(offset).all() or np.any(np.abs(offset) > 1e6):
        return None
    return [float(value) for value in offset]
