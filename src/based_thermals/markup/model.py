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
from .tables import cells

KINDS = ("text", "textbox", "rectangle", "ellipse", "polygon", "line", "arrow", "table")
TITLES = ("Label", "Text box", "Rectangle", "Ellipse", "Polygon", "Line", "Arrow", "Table")


def annotation(kind: str) -> dict[str, Any]:
    """
    Create an unplaced annotation with independently editable border, fill and text styles.
    """
    if kind not in KINDS:
        raise ValueError("Choose a Markup drawing tool")
    result: dict[str, Any] = {
        "id": uuid4().hex, "kind": kind, "points": [], "z_index": 0,
        "text": ("Label" if kind == "text" else "Text box" if kind == "textbox"
                 else "Name\tValue\nItem\t--" if kind == "table" else ""),
        "bindings": {},
        "border_color": "dynamic", "border_opacity": 0.0 if kind == "text" else 1.0, "width": 1,
        "fill": kind in ("textbox", "table"), "fill_color": "#000000", "fill_opacity": 0.4,
        "text_color": "dynamic", "text_opacity": 1.0, "font_size": 16,
    }
    if kind == "text":
        result["text_offset"] = [0., 0.]
    return result


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
        if (not isinstance(item, dict) or not _FIELDS - {"z_index", "bindings", "text_offset"} <= set(item)
                or not set(item) <= _FIELDS):
            raise ValueError("Unknown or missing Markup annotation field")
        item.setdefault("z_index", 0)
        offset = item.get("text_offset")
        if "text_offset" in item and (
            not isinstance(offset, list) or len(offset) != 2
            or any(type(value) not in (int, float) or not -1e6 <= value <= 1e6
                   or not math.isfinite(value) for value in offset)
        ):
            raise ValueError("Markup text offset must contain two finite bounded coordinates")
        bindings = item.setdefault("bindings", {})
        if (not isinstance(bindings, dict) or len(bindings) > 128
                or any(not isinstance(key, str) or not 1 <= len(key) <= 160
                       or any(c in key for c in "[]\n")
                       or not isinstance(value, str) or not 1 <= len(value) <= 512
                       for key, value in bindings.items())):
            raise ValueError("Invalid Markup variable bindings")
        if type(item["z_index"]) is not int or not -1000 <= item["z_index"] <= 1000:
            raise ValueError("Markup Z index must be an integer from -1000 to 1000")
        identity = item["id"]
        if (not isinstance(identity, str) or not identity or len(identity) > 64
                or identity in identities):
            raise ValueError("Markup annotation identities must be unique")
        identities.add(identity)
        if item["kind"] not in KINDS or not isinstance(item["text"], str) or len(item["text"]) > 8192:
            raise ValueError("Invalid Markup kind or label text")
        if item["text"].count("{{") > 128:
            raise ValueError("Markup text supports up to 128 readouts")
        if item["kind"] == "table":
            cells(item["text"])
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
