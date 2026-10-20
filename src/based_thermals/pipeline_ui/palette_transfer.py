"""
Palette library JSON transfers and active pipeline snapshot updates.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..custom_nodes.configuration import read_controls
from ..palettes import is_palette_choice, read_library, save_palettes, write_library
from ..pipeline import software_tabs

if TYPE_CHECKING:
    from .editor import PipelineEditor
    from .palette_builder import PaletteBuilder


def import_palettes(builder: PaletteBuilder) -> None:
    """
    Validate and merge a complete import, asking before replacing colliding names.
    """
    if builder.locked:
        return
    path, _ = QFileDialog.getOpenFileName(builder, "Import palettes", "", "Palette JSON (*.json)")
    if not path or builder.locked:
        return
    try:
        incoming = read_library(Path(path))
        if not incoming:
            raise ValueError("The file contains no palettes")
        collisions = {name for name in builder.palettes
                      if name.casefold() in {key.casefold() for key in incoming}}
        if collisions and QMessageBox.question(
            builder, "Replace palettes", "Replace these saved palettes?\n" + "\n".join(sorted(collisions)),
        ) != QMessageBox.StandardButton.Yes:
            return
        if not builder.discard_draft():
            return
        candidate = {name: value for name, value in builder.palettes.items() if name not in collisions}
        candidate.update(incoming)
        save_palettes(candidate)
        builder.palettes = candidate
        builder.load_draft(next(iter(incoming)))
        builder.changed(None, None)
    except (OSError, ValueError, UnicodeError) as exc:
        QMessageBox.warning(builder, "Could not import palettes", str(exc))


def export_palette(builder: PaletteBuilder) -> None:
    """
    Export the current draft, including pending edits, without implicitly saving it locally.
    """
    if builder.locked:
        return
    try:
        builder.commit_pending_inputs()
        value = builder.draft()
        path, _ = QFileDialog.getSaveFileName(
            builder, "Export palette", "palette.json", "Palette JSON (*.json)",
        )
        if path and not builder.locked:
            write_library(Path(path), {value["name"]: value})
    except (OSError, ValueError) as exc:
        QMessageBox.warning(builder, "Could not export palette", str(exc))


def refresh_pipeline_palettes(editor: PipelineEditor, old: str | None, new: str | None) -> None:
    """
    Validate all active replacements before committing any live parameter changes.
    """
    if editor.locked:
        return
    candidate = deepcopy(editor.document)
    current_tabs = software_tabs(editor.document)
    updates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for tab, nodes in software_tabs(candidate).items():
        for index, item in enumerate(nodes):
            params = item["params"]
            changed = False
            if old is not None and new is not None and old != new:
                if item["type"] == "colors" and params["palette"] == old:
                    params["palette"] = new
                    changed = True
                elif item["type"] == "custom":
                    config = json.loads(params["config"])
                    fields = read_controls(params.get("controls", "[]"), config)
                    keys = {field["key"] for field in fields
                            if field["type"] == "palette" or is_palette_choice(field)}
                    # Preserve JSON-only packages that used the original shared helper.
                    keys.add("palette")
                    updated = False
                    for key in keys:
                        if config.get(key) == old:
                            config[key] = new
                            updated = True
                    if updated:
                        params["config"] = json.dumps(config, allow_nan=False)
                        changed = True
            if changed:
                updates.append((current_tabs[tab][index]["params"], params))
    if updates:
        editor.validate_document(candidate)
        for current, replacement in updates:
            current.update(replacement)
    editor.rebuild(software_only=True)
    if updates:
        editor.publish()
