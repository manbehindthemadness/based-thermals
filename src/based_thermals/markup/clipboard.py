"""
Bounded portable clipboard objects shared by the viewer and Imaging processes.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from .model import serialize

FORMAT = "based-thermals-markup"


def encode(item: dict[str, Any]) -> str:
    """
    Copy one complete annotation, including variable bindings and appearance.
    """
    serialize([item])
    return json.dumps({"format": FORMAT, "version": 1, "annotation": item}, ensure_ascii=False)


def decode(text: object) -> dict[str, Any]:
    """
    Validate external clipboard text without evaluating expressions or importing code.
    """
    if not isinstance(text, str) or len(text) > 70000:
        raise ValueError("Clipboard does not contain a Markup object")
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Clipboard does not contain a Markup object") from exc
    if (not isinstance(payload, dict) or payload.get("format") != FORMAT
            or type(payload.get("version")) is not int or payload["version"] != 1
            or not isinstance(payload.get("annotation"), dict)):
        raise ValueError("Clipboard does not contain a Markup object")
    try:
        item = deepcopy(payload["annotation"])
        serialize([item])
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Clipboard contains an invalid Markup object") from exc
    return item


def paste(items: list[dict[str, Any]], text: object) -> dict[str, Any]:
    """
    Append a fresh identity with a small bounded offset, preserving the original shape.
    """
    if len(items) >= 128:
        raise ValueError("A Markup node supports up to 128 annotations")
    item = decode(text)
    item["id"] = uuid4().hex
    if item["points"]:
        for axis in (0, 1):
            maximum = float(max(point[axis] for point in item["points"]))
            minimum = float(min(point[axis] for point in item["points"]))
            offset = min(.02, 1 - maximum) if maximum < 1 else -min(.02, minimum)
            for point in item["points"]:
                point[axis] += offset
    serialize([*items, item])
    items.append(item)
    return item
