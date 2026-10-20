"""
Live application palette picker for declarative Custom controls.
"""

from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox

from ..custom_nodes.palettes import palette_choices, palette_key
from ..palettes import PREFIX, read_palette


class PalettePicker(QComboBox):
    """
    Discover app palettes when opened and keep active popup selection through frame refreshes.
    """

    def __init__(self, value: str, changed: Callable[[str], None]) -> None:
        """
        Display friendly labels while preserving portable identifiers in the node config.
        """
        super().__init__()
        self.changed = changed
        self.value = value
        self.setMinimumWidth(1)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.refresh(value, discover=True)
        self.activated.connect(self.select)

    def refresh(self, value: str, *, discover: bool = False) -> None:
        """
        Apply accepted state silently without scanning disk or rebuilding an open popup.
        """
        if self.view().isVisible() and value == self.value:
            return
        self.value = value
        key = palette_key(value)
        with QSignalBlocker(self):
            if discover:
                self.clear()
                for token, label in palette_choices():
                    self.addItem(label, token)
            if self.findData(key) < 0 and key.startswith(PREFIX):
                self.addItem(f"Embedded · {read_palette(key)['name']}", key)
            self.setCurrentIndex(self.findData(key))

    def showPopup(self) -> None:
        """
        Discover new library entries on the next opening, retaining the current snapshot.
        """
        self.refresh(self.value, discover=True)
        super().showPopup()

    def select(self, _index: int) -> None:
        """
        Publish a user selection once, leaving routine state refreshes read-only.
        """
        value = self.currentData()
        if self.isEnabled() and isinstance(value, str):
            self.value = value
            self.changed(value)
