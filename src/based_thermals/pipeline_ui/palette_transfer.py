"""
Palette library JSON transfers and active pipeline snapshot updates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..palettes import read_library, save_palettes, write_library
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
    if not path:
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
        if path:
            write_library(Path(path), {value["name"]: value})
    except (OSError, ValueError) as exc:
        QMessageBox.warning(builder, "Could not export palette", str(exc))


def refresh_pipeline_palettes(editor: PipelineEditor, old: str | None, new: str | None) -> None:
    """
    Update exact active snapshots on Save and refresh menus without changing saved presets.
    """
    changed = False
    if editor.locked:
        return
    for nodes in software_tabs(editor.document).values():
        for item in nodes:
            params = item["params"]
            if old is not None and new is not None and old != new:
                if item["type"] == "colors" and params["palette"] == old:
                    params["palette"] = new
                    changed = True
                elif item["type"] == "custom":
                    config = json.loads(params["config"])
                    if config.get("palette") == old:
                        config["palette"] = new
                        params["config"] = json.dumps(config, allow_nan=False)
                        changed = True
    editor.rebuild(software_only=True)
    if changed:
        editor.publish()
