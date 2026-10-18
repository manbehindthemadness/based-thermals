"""
Classify display-only visualizations without changing sensor measurement validity or activation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from typing import Any

from ..custom_nodes.bundle import checked_files
from ..custom_nodes.metadata import embedded_metadata
from .document import software_tabs

# Recognize the originally shipped Thermal Surface entry point in existing saved packages.
_LEGACY_SURFACE = "57615dbc88132f8757594bc9ed82c7246866e7cc7e431f70dd70f09d2a7b57e3"


@lru_cache(maxsize=32)
def image_feed(package: str) -> bool:
    """
    Read literal output metadata without executing code, including released height-map snapshots.
    """
    source = next((source for name, source in checked_files(package) if name == "__init__.py"), "")
    if not source:
        return True
    declared = embedded_metadata(source).get("OUTPUT_KIND")
    if declared is not None:
        return json.loads(declared) == "image"
    return sha256(source.encode("utf-8")).hexdigest() != _LEGACY_SURFACE


def measurement_graphics_visible(document: dict[str, Any] | None) -> bool:
    """
    Follow contributing image branches, excluding idle previews, bypasses and mask-only inputs.

    A visualization hides native graphics across the composite. A later full, unmasked
    replacement by an image feed restores them; partial replacements retain the base policy.
    """
    if document is None:
        return True
    tabs = software_tabs(document)
    completed: dict[str, tuple[bool, bool]] = {}

    def branch(tab: str) -> tuple[bool, bool]:
        """
        Return whether a branch remains image-based and occupies its full output canvas.
        """
        if tab in completed:
            return completed[tab]
        feed, full = True, True
        for item in tabs[tab]:
            if item["bypass"]:
                continue
            kind, params = item["type"], item["params"]
            if kind == "custom":
                feed &= image_feed(params["package"])
            elif kind == "transform":
                full &= ((params["resize_mode"] == "resolution"
                          or params["width"] == params["height"] == 100)
                         and all(params[key] == 0 for key in (
                             "position_x", "position_y", "skew_x", "skew_y", "top_left_x",
                             "top_left_y", "top_right_x", "top_right_y", "bottom_right_x",
                             "bottom_right_y", "bottom_left_x", "bottom_left_y",
                         )))
            elif kind == "combine":
                mode = params["mode"]
                if mode == "mask" or mode != "replace" and params["opacity"] == 0:
                    continue
                incoming, incoming_full = branch(params["tab"]) if params["tab"] in tabs else (True, True)
                unmasked = params["mask_source"] == "none"
                if unmasked and (mode == "replace" and incoming_full
                                 or mode == "opacity" and params["opacity"] == 1):
                    feed, full = incoming, incoming_full
                else:
                    feed &= incoming
                    full |= incoming_full and unmasked
        classification = bool(feed), bool(full)
        completed[tab] = classification
        return classification

    visible, _full = branch("A")
    return visible
