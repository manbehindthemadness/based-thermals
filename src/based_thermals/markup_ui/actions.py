"""
Shared native object clipboard shortcuts and nonblocking context menus.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QAction, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from ..markup.clipboard import decode


def clipboard_text() -> str:
    """
    Read bounded portable object text from the system clipboard.
    """
    text = QApplication.clipboard().text()
    return text if len(text) <= 70000 else ""


class ObjectActions(QObject):
    """
    Keep object shortcuts scoped to the canvas or list, leaving text fields untouched.
    """

    def __init__(self, widget: QWidget, run: Callable[[str, str], None],
                 available: Callable[[], tuple[bool, bool]],
                 suspended: Callable[[], bool] = lambda: False) -> None:
        """
        Attach keyboard handling to one object surface with dynamic lock checks.
        """
        super().__init__(widget)
        self.widget, self.run, self.available = widget, run, available
        self.suspended = suspended
        self.menu: QMenu | None = None
        widget.installEventFilter(self)

    def invoke(self, operation: str) -> None:
        """
        Recheck locks when executing a menu or shortcut, including an already open menu.
        """
        selected, writable = self.available()
        if not writable or operation != "paste" and not selected:
            return
        self.run(operation, clipboard_text() if operation == "paste" else "")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Consume only object clipboard/delete shortcuts on the attached object surface.
        """
        if (watched is self.widget and isinstance(event, QKeyEvent)
                and event.type() == QEvent.Type.KeyPress):
            operation = ("copy" if event.matches(QKeySequence.StandardKey.Copy)
                         else "paste" if event.matches(QKeySequence.StandardKey.Paste)
                         else "delete" if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                         and event.modifiers() == Qt.KeyboardModifier.NoModifier
                         and not self.suspended() else "")
            if operation:
                self.invoke(operation)
                return True
        return super().eventFilter(watched, event)

    def popup(self, position: QPoint) -> None:
        """
        Open a native menu without blocking frame delivery or creating nested event loops.
        """
        if self.menu is not None:
            self.menu.close()
            self.menu.deleteLater()
        menu = QMenu(self.widget)
        self.menu = menu
        selected, writable = self.available()
        valid_clipboard = False
        try:
            decode(clipboard_text())
            valid_clipboard = True
        except ValueError:
            pass
        for title, operation, enabled in (
            ("Edit in Imaging…", "edit", selected and writable),
            ("Copy", "copy", selected and writable),
            ("Paste", "paste", writable and valid_clipboard),
            ("Delete", "delete", selected and writable),
        ):
            action = QAction(title, menu)
            menu.addAction(action)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False, op=operation: self.invoke(op))
        menu.popup(position)
