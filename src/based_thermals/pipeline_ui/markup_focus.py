"""
One-shot navigation from a displayed annotation into its owning pipeline controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import QScrollArea

from ..pipeline import software_tabs

if TYPE_CHECKING:
    from ..view_window import ViewWindow


def focus_markup(window: ViewWindow, request: Any) -> None:
    """
    Expand the correct branch/node, select its stable element identity and reveal its editor.
    """
    if not isinstance(request, dict) or not isinstance(request.get("serial"), str):
        return
    if request["serial"] == getattr(window, "_markup_focus_serial", None):
        return
    window._markup_focus_serial = request["serial"]
    editor = window.pipeline_editor
    target = next(((tab, item) for tab, nodes in software_tabs(editor.document).items()
                   for item in nodes if item["id"] == request.get("node") and item["type"] == "markup"), None)
    if target is None:
        return
    tab, item = target
    item["expanded"] = True
    editor.tab = tab
    editor.set_software_expanded(True)
    editor.rebuild(software_only=True)
    controls = editor.markup_widgets[item["id"]]
    controls.text.clearFocus()
    blocker = QSignalBlocker(controls.objects)
    controls.objects.setCurrentRow(controls.index_of(request.get("annotation")))
    blocker.unblock()
    controls.refresh()

    def reveal() -> None:
        """
        Scroll after Qt sizes the expanded branch and focus only this explicit edit request.
        """
        if editor.markup_widgets.get(item["id"]) is not controls:
            return
        window.activateWindow()
        field = controls.text if controls.text.isEnabled() else controls.objects
        field.setFocus(Qt.FocusReason.OtherFocusReason)
        parent = controls.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(controls.text, 12, 12)
            parent = parent.parentWidget()

    QTimer.singleShot(0, reveal)
