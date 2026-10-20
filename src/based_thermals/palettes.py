"""
Validated, portable custom gradients and the user's palette library.
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .preference_io import save_json
from .window_preferences import _path

PREFIX = "custom:"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_STOPS = 256


def validate_palette(value: Any) -> dict[str, Any]:
    """
    Normalize a named gradient with strictly increasing, endpoint-inclusive RGB stops.
    """
    if not isinstance(value, dict) or set(value) != {"name", "stops"}:
        raise ValueError("A palette requires name and stops")
    name, stops = value["name"], value["stops"]
    if (not isinstance(name, str) or not name.strip() or len(name) > 100
            or any(ord(char) < 32 for char in name)):
        raise ValueError("Palette names must contain 1–100 printable characters")
    if not isinstance(stops, list) or not 2 <= len(stops) <= MAX_STOPS:
        raise ValueError(f"A palette requires 2–{MAX_STOPS} color stops")
    normalized = []
    previous = -1.0
    for stop in stops:
        if not isinstance(stop, dict) or set(stop) != {"position", "color"}:
            raise ValueError("Each stop requires position and color")
        position, color = stop["position"], stop["color"]
        if (isinstance(position, bool) or not isinstance(position, (int, float))
                or not 0 <= position <= 1 or not math.isfinite(position)
                or position <= previous):
            raise ValueError("Stop positions must increase strictly from 0 to 1")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Stop colors must be #RRGGBB")
        normalized.append({"position": float(position), "color": color.lower()})
        previous = position
    if normalized[0]["position"] != 0 or normalized[-1]["position"] != 1:
        raise ValueError("The first and last stop must be at 0% and 100%")
    return {"name": name.strip(), "stops": normalized}


def palette_token(value: Any) -> str:
    """
    Embed a complete palette in a string-valued pipeline parameter for portable saves.
    """
    result = PREFIX + json.dumps(validate_palette(value), separators=(",", ":"), ensure_ascii=True)
    return result


@lru_cache(maxsize=128)
def _read_token(value: str) -> dict[str, Any]:
    """
    Parse bounded embedded definitions once, independently of the user's library.
    """
    if len(value) > 32768 or not value.startswith(PREFIX):
        raise ValueError("Invalid custom palette")
    result = validate_palette(json.loads(value[len(PREFIX):]))
    return result


def read_palette(value: str) -> dict[str, Any]:
    """
    Return a detached definition so UI edits cannot mutate cached pipeline snapshots.
    """
    cached = _read_token(value)
    return {"name": cached["name"], "stops": [dict(stop) for stop in cached["stops"]]}


@lru_cache(maxsize=64)
def custom_lut(value: str) -> np.ndarray:
    """
    Build a small immutable 256×3 BGR table using piecewise linear RGB interpolation.
    """
    stops = _read_token(value)["stops"]
    positions = [stop["position"] for stop in stops]
    colors = [[int(stop["color"][offset:offset + 2], 16) for offset in (5, 3, 1)]
              for stop in stops]
    channels = [np.interp(np.linspace(0, 1, 256), positions, np.asarray(colors)[:, channel])
                for channel in range(3)]
    result = np.stack(channels, axis=1).round().astype(np.uint8)
    result.setflags(write=False)
    return result


def read_library(path: Path) -> dict[str, dict[str, Any]]:
    """
    Validate an entire versioned import before changing any saved palettes.
    """
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Palette files must be smaller than 8 MiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or set(document) != {"version", "palettes"}
            or type(document["version"]) is not int or document["version"] != 1
            or not isinstance(document["palettes"], list)
            or len(document["palettes"]) > 128):
        raise ValueError("Expected a version 1 palette library with at most 128 palettes")
    result: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    for entry in document["palettes"]:
        palette = validate_palette(entry)
        if palette["name"].casefold() in names:
            raise ValueError("Palette names must be unique (ignoring case)")
        names.add(palette["name"].casefold())
        result[palette["name"]] = palette
    return result


def write_library(path: Path, palettes: dict[str, dict[str, Any]]) -> None:
    """
    Atomically save a bounded, validated library or export without touching other settings.
    """
    if len(palettes) > 128:
        raise ValueError("At most 128 custom palettes can be saved")
    entries = [validate_palette(palettes[name]) for name in sorted(palettes, key=str.casefold)]
    if (any(name != value["name"] for name, value in palettes.items())
            or len({entry["name"].casefold() for entry in entries}) != len(entries)):
        raise ValueError("Palette names must match their unique library names")
    save_json(path, {"version": 1, "palettes": entries}, indent=2)


def load_palettes() -> dict[str, dict[str, Any]]:
    """
    Restore the local library, treating missing or malformed preferences as empty.
    """
    try:
        result = read_library(_path().with_name("palettes.json"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return result


def save_palettes(palettes: dict[str, dict[str, Any]]) -> None:
    """
    Save custom palettes beside the other application preferences.
    """
    write_library(_path().with_name("palettes.json"), palettes)


def is_palette_choice(control: dict[str, Any]) -> bool:
    """
    Recognize the shared visualization palette menu in existing embedded packages.
    """
    if control.get("type") != "choice" or control.get("key") != "palette":
        return False
    from .custom_nodes.visualization import palette_names

    return control.get("options") == list(palette_names())
