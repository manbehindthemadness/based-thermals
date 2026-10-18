"""
Bounded tab-separated table content and cached OpenCV grid/text masks.
"""

from typing import Any

import cv2
import numpy as np


def cells(text: str) -> list[list[str]]:
    """
    Parse portable plain-text cells without HTML, code execution or spreadsheet dependencies.
    """
    rows = [line.split("\t") for line in text.split("\n")]
    columns = max(len(row) for row in rows)
    if len(rows) > 32 or columns > 16:
        raise ValueError("Markup tables support up to 32 rows and 16 columns")
    return [row + [""] * (columns - len(row)) for row in rows]


def paint_table(border: np.ndarray, text: np.ndarray, item: dict[str, Any],
                matrix: tuple[float, ...], shape: tuple[int, int], offset: tuple[int, int]) -> None:
    """
    Rasterize transformed cell borders and clipped upright text only within each cell's ROI.
    """
    from .paint import project

    rows = cells(item["text"])
    height, width = shape
    low, high = np.min(item["points"], axis=0), np.max(item["points"], axis=0)
    row_count, column_count = len(rows), len(rows[0])
    font = item["font_size"] / 22
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            left = low[0] + (high[0] - low[0]) * column / column_count
            right = low[0] + (high[0] - low[0]) * (column + 1) / column_count
            top = low[1] + (high[1] - low[1]) * row_index / row_count
            bottom = low[1] + (high[1] - low[1]) * (row_index + 1) / row_count
            corners = project([[left, top], [right, top], [right, bottom], [left, bottom]], matrix)
            if not np.isfinite(corners).all():
                continue
            local = np.clip(np.rint(corners * (width - 1, height - 1) - offset), -65536, 65536).astype(np.int32)
            cv2.polylines(border, [local], True, 255, item["width"], cv2.LINE_AA)
            x0, y0 = np.maximum(local.min(axis=0), 0)
            x1, y1 = np.minimum(local.max(axis=0) + 1, (text.shape[1], text.shape[0]))
            if x0 >= x1 or y0 >= y1 or not value:
                continue
            clip = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(clip, [local - (x0, y0)], 255)
            glyph = np.zeros_like(clip)
            cv2.putText(glyph, value, (6, item["font_size"] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        font, 255, 1, cv2.LINE_AA)
            cv2.bitwise_and(glyph, clip, dst=glyph)
            region = text[y0:y1, x0:x1]
            cv2.max(region, glyph, dst=region)
