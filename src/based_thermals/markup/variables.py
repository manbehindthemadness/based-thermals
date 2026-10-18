"""
Discoverable readout catalog and stable JSON-pointer references to pipeline settings.
"""

from __future__ import annotations

import json
import math
from typing import Any

CORE = {
    "Temperature.Minimum": ("Temperatures", "Minimum", "temperature"),
    "Temperature.Maximum": ("Temperatures", "Maximum", "temperature"),
    "Temperature.Average": ("Temperatures", "Average", "temperature"),
    "Temperature.Center": ("Temperatures", "Camera center", "temperature"),
    "Measurement.Valid": ("Measurements", "Readings valid", ""),
    "Measurement.Status": ("Measurements", "Reading status", ""),
    "Temperature.Unit": ("Temperatures", "Display unit", ""),
    "Temperature.Valid": ("Temperatures", "Temperature data valid", ""),
    "Capture.FPS": ("Frame rates and counters", "Received frame rate", "fps"),
    "Capture.Frames": ("Frame rates and counters", "Received frame counter", "frames"),
    "Camera.NominalFPS": ("Frame rates and counters", "Camera nominal rate", "fps"),
    "Display.FPS": ("Frame rates and counters", "Painted frame rate", "fps"),
    "Display.Frames": ("Frame rates and counters", "Painted frame counter", "frames"),
    "Timing.Pipeline": ("Timings", "Pipeline processing", "ms"),
    "Timing.Draw": ("Timings", "Viewer draw (10-frame average)", "ms"),
    "Timing.Total": ("Timings", "Capture to paint (10-frame average)", "ms"),
    "Session.Elapsed": ("Timers", "Session elapsed", "s"),
    "Session.Timer": ("Timers", "Session timer (HH:MM:SS)", ""),
    "Clock.Time": ("Timers", "Local time", ""),
    "Clock.Date": ("Timers", "Local date", ""),
    "Recording.Elapsed": ("Recording", "Recording elapsed", "s"),
    "Recording.Frames": ("Recording", "Recorded frame counter", "frames"),
}


def pipeline_nodes(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """
    Enumerate saved hardware and software nodes without importing UI or executing package code.
    """
    from ..pipeline import software_tabs

    nodes = [(f"Pipeline {tab}", node) for tab, items in software_tabs(document).items() for node in items]
    nodes.extend(("Camera hardware", node) for node in document.get("hardware", []))
    nodes.extend(("Visible hardware", node) for node in document.get("visible_hardware", []))
    return nodes


def escaped(part: str) -> str:
    """
    Escape JSON-pointer components without making assumptions about Custom configuration keys.
    """
    return part.replace("~", "~0").replace("/", "~1")


def leaves(value: Any, path: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    """
    Expose bounded scalar JSON settings, omitting embedded source, annotation and package blobs.
    """
    if depth > 6:
        return []
    if isinstance(value, dict):
        result = []
        for key, item in list(value.items())[:256]:
            if key not in ("annotations", "files", "code", "controls"):
                if key == "config" and isinstance(item, str) and len(item) <= 65536:
                    try:
                        item = json.loads(item)
                    except (ValueError, RecursionError):
                        pass
                result.extend(leaves(item, f"{path}/{escaped(key)}", depth + 1))
        return result
    if isinstance(value, list):
        return [leaf for index, item in enumerate(value[:64])
                for leaf in leaves(item, f"{path}/{index}", depth + 1)]
    if value is None or type(value) in (bool, int, float) or isinstance(value, str) and len(value) <= 256:
        return [(path, value)]
    return []


def pipeline_value(document: dict[str, Any], reference: str) -> Any:
    """
    Resolve a stable node identity and JSON path after tab moves, renames and reordering.
    """
    parts = reference.split("/")
    if len(parts) < 3 or parts[0] != "Pipeline":
        return None
    current = next((node for _, node in pipeline_nodes(document) if node["id"] == parts[1]), None)
    for part in parts[2:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, str) and len(current) <= 65536:
            try:
                current = json.loads(current)
            except (ValueError, RecursionError):
                return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return finite(current)


def catalog(document: dict[str, Any], live: dict[str, Any], measurements: list[dict]) -> list[dict]:
    """
    Return searchable human labels plus canonical identities and snapshot values for the picker.
    """
    from ..pipeline import CATALOG

    unit = live.get("Temperature.Unit", "C")
    result = [{"key": key, "alias": key, "group": group, "label": title,
               "unit": unit if measure == "temperature" else measure, "value": live.get(key)}
              for key, (group, title, measure) in CORE.items()]
    for measurement in measurements:
        number, name = measurement["number"], measurement["name"]
        name = str(name).replace("[", "(").replace("]", ")").replace("\n", " ")[:64]
        for field in ("Minimum", "Maximum", "Average", "Count"):
            key = f"Measurement.{number}.{field}"
            result.append({"key": key, "alias": f"Measurement {number} ({name}).{field}",
                           "group": f"Measurement {number}: {name}", "label": field,
                           "unit": "pixels" if field == "Count" else unit, "value": live.get(key)})
    counts: dict[tuple[str, str], int] = {}
    for group, node in pipeline_nodes(document):
        kind = node["type"]
        stack = "hardware" if group == "Camera hardware" else "software"
        title, definitions = CATALOG.get(stack, {}).get(kind, (kind.replace("_", " ").title(), {}))
        counts[group, kind] = counts.get((group, kind), 0) + 1
        name = f"{title} {counts[group, kind]}"
        for path, value in [("/bypass", node["bypass"]), *leaves(node["params"], "/params")]:
            field = path.rsplit("/", 1)[-1]
            spec = definitions.get(field) if path == f"/params/{field}" else None
            label = spec.title if spec is not None else field.replace("_", " ").title()
            result.append({"key": f"Pipeline/{node['id']}{path}",
                           "alias": f"{group} / {name} / {path.removeprefix('/params/').replace('/', ' / ')}"[:160],
                           "group": f"{group} / {name}", "label": label,
                           "unit": spec.unit if spec is not None else "", "value": finite(value)})
    return result


def finite(value: Any) -> float | int | str | bool | None:
    """
    Keep transport snapshots JSON-safe when a measurement or runtime statistic is unavailable.
    """
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if len(value) <= 256 else None
    if type(value) is int and abs(value) > 1e15:
        return None
    return value if value is None or type(value) in (int, str, bool) else None


def references(document: dict[str, Any]) -> set[str]:
    """
    Gather only active annotations' dependencies, resolving readable aliases to stable identities.
    """
    from .expressions import REFERENCE, TOKEN
    from .model import read_annotations

    required = set()
    for _, node in pipeline_nodes(document):
        if node["type"] == "markup" and not node["bypass"]:
            for item in read_annotations(node["params"]["annotations"]):
                for token in TOKEN.finditer(item["text"]):
                    required.update(item["bindings"].get(match.group(1), match.group(1))
                                    for match in REFERENCE.finditer(token.group(1)))
    return required
