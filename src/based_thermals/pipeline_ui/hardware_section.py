"""
Persist camera hardware disclosure state independently of pipeline documents.
"""

from typing import cast

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QPushButton, QWidget


class HardwareSection(QPushButton):
    """
    Match the Environment disclosure while respecting camera capability availability.
    """

    def __init__(
        self, title: str, content: QWidget, settings_key: str, *, available: bool = True,
    ) -> None:
        """
        Restore an initially collapsed, independently remembered camera section.
        """
        super().__init__()
        self._title = title
        self._content = content
        self._settings_key = settings_key
        self._available = available
        self._settings = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "based-thermals", "desktop",
        )
        self.setAccessibleName(title.split(" ·", 1)[0])
        self.setStyleSheet("text-align: left; font-weight: bold")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCheckable(True)
        self.setChecked(cast(bool, self._settings.value(settings_key, False, type=bool)))
        self.toggled.connect(self._set_expanded)
        self.set_available(available)

    def _set_expanded(self, expanded: bool) -> None:
        """
        Remember only presentation state, without emitting a pipeline or hardware change.
        """
        self._settings.setValue(self._settings_key, expanded)
        self.set_available(self._available)

    def set_available(self, available: bool) -> None:
        """
        Hide unsupported cameras without losing their remembered disclosure selection.
        """
        self._available = available
        self.setVisible(available)
        self._content.setVisible(available and self.isChecked())
        self.setText(("▾ " if self.isChecked() else "▸ ") + self._title)
