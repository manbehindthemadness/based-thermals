"""
Portable, bounded annotation geometry independent of sensor measurements.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any
from uuid import uuid4

from ..custom_nodes.colors import color_value

KINDS = ("text", "textbox", "rectangle", "ellipse", "polygon", "line", "arrow")
TITLES = ("Label", "Text box", "Rectangle", "Ellipse", "Polygon", "Line", "Arrow")


def annotation(kind: str) -> dict[str, Any]:
    """
    Create an unplaced annotation with independently editable border, fill and text styles.
    """
    if kind not in KINDS:
        raise ValueError("Choose a Markup drawing tool")
    return {
        "id": uuid4().hex, "kind": kind, "points": [], "z_index": 0,
        "text": "Label" if kind == "text" else "Text box" if kind == "textbox" else "",
        "border_color": "dynamic", "border_opacity": 0.0 if kind == "text" else 1.0, "width": 1,
        "fill": kind == "textbox", "fill_color": "#000000", "fill_opacity": 0.4,
        "text_color": "dynamic", "text_opacity": 1.0, "font_size": 16,
    }


_FIELDS = frozenset(annotation("text"))


@lru_cache(maxsize=32)
def read_annotations(text: str) -> tuple[dict[str, Any], ...]:
    """
    Validate serialized annotations without importing code or interpreting temperature values.

    Cached results are read-only by convention; editors copy them before mutation.
    """
    if not isinstance(text, str) or len(text.encode("utf-8")) > 65536:
        raise ValueError("Markup annotations must fit within 64 KiB")
    try:
        items = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid Markup annotation JSON") from exc
    if not isinstance(items, list) or len(items) > 128:
        raise ValueError("A Markup node supports up to 128 annotations")
    identities = set()
    for item in items:
        if not isinstance(item, dict) or set(item) not in (_FIELDS, _FIELDS - {"z_index"}):
            raise ValueError("Unknown or missing Markup annotation field")
        item.setdefault("z_index", 0)
        if type(item["z_index"]) is not int or not -1000 <= item["z_index"] <= 1000:
            raise ValueError("Markup Z index must be an integer from -1000 to 1000")
        identity = item["id"]
        if (not isinstance(identity, str) or not identity or len(identity) > 64
                or identity in identities):
            raise ValueError("Markup annotation identities must be unique")
        identities.add(identity)
        if item["kind"] not in KINDS or not isinstance(item["text"], str) or len(item["text"]) > 1024:
            raise ValueError("Invalid Markup kind or label text")
        if type(item["fill"]) is not bool:
            raise ValueError("Markup fill must be a boolean")
        for key in ("border_color", "fill_color", "text_color"):
            color_value(item[key])
        for key in ("border_opacity", "fill_opacity", "text_opacity"):
            value = item[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Markup opacity must be between 0 and 1")
        for key, low, high in (("width", 1, 12), ("font_size", 8, 96)):
            if type(item[key]) is not int or not low <= item[key] <= high:
                raise ValueError(f"Markup {key} must be an integer from {low} to {high}")
        points = item["points"]
        count = 1 if item["kind"] == "text" else 2
        if (not isinstance(points, list) or len(points) > 64
                or points and (len(points) < 3 if item["kind"] == "polygon" else len(points) != count)):
            raise ValueError("Invalid Markup geometry")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                           for v in point)):
                raise ValueError("Markup points must be finite normalized image coordinates")
    return tuple(items)


def serialize(items: list[dict[str, Any]]) -> str:
    """
    Save validated current annotation definitions using portable JSON.
    """
    result = json.dumps(items, ensure_ascii=False, allow_nan=False)
    read_annotations(result)
    return result
