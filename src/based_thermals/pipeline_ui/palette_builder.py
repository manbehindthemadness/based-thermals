"""
Collapsible custom palette library with editable stops and portable JSON transfers.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, QSignalBlocker
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..palette_templates import builtin_template
from ..palettes import (
    MAX_STOPS,
    custom_lut,
    load_palettes,
    palette_token,
    save_palettes,
    validate_palette,
)
from ..pipeline_model.catalog import PALETTES
from .hardware_section import HardwareSection


class PaletteBuilder(QWidget):
    """
    Edit an explicit draft without altering active pipelines until the user saves it.
    """

    def __init__(self, changed: Callable[[str | None, str | None], None]) -> None:
        """
        Restore the palette library and remembered disclosure state below imaging controls.
        """
        super().__init__()
        self.changed = changed
        self.palettes = load_palettes()
        self.current: str | None = None
        self.source_palette: str | None = None
        self.baseline = ""
        self.locked = False
        self.building = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.content = QWidget()
        self.section = HardwareSection("Palette builder", self.content, "camera/palettes_expanded")
        layout.addWidget(self.section)
        layout.addWidget(self.content)
        body = QVBoxLayout(self.content)
        body.setContentsMargins(12, 0, 0, 0)
        explanation = QLabel(
            "Start from a software palette or build a gradient from cool (0%) to hot (100%). "
            "Save an editable copy for use in App colors "
            "and thermal visualization nodes. Camera hardware palettes are unchanged."
        )
        explanation.setWordWrap(True)
        body.addWidget(explanation)
        self.combo = QComboBox()
        self.combo.setAccessibleName("Software and custom palettes")
        self.combo.activated.connect(self.select)
        body.addWidget(self.combo)
        self.name = QLineEdit()
        self.name.setMaxLength(100)
        self.name.setPlaceholderText("Palette name")
        self.name.setAccessibleName("Palette name")
        self.name.textChanged.connect(self.update_preview)
        body.addWidget(self.name)
        self.preview = QLabel()
        self.preview.setMinimumHeight(28)
        self.preview.setScaledContents(True)
        self.preview.setAccessibleName("Palette preview, cool to hot")
        body.addWidget(self.preview)
        self.stops = QTableWidget(0, 2)
        self.stops.setHorizontalHeaderLabels(["Position (%)", "Color"])
        self.stops.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.stops.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stops.setMinimumHeight(150)
        self.stops.setMaximumHeight(240)
        body.addWidget(self.stops)
        stop_buttons = QHBoxLayout()
        for title, callback in (("Add stop", self.add_stop), ("Remove stop", self.remove_stop)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            stop_buttons.addWidget(button)
        body.addLayout(stop_buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        body.addWidget(self.status)
        buttons = QHBoxLayout()
        from .palette_transfer import export_palette, import_palettes

        for title, callback in (
            ("New", self.new), ("Save", self.save), ("Delete", self.delete),
            ("Import…", lambda: import_palettes(self)),
            ("Export…", lambda: export_palette(self)),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        body.addLayout(buttons)
        self.refresh_library()
        self.load_draft(None)

    def refresh_library(self) -> None:
        """
        Rebuild the saved listing without implicitly discarding a draft.
        """
        with QSignalBlocker(self.combo):
            self.combo.clear()
            self.combo.addItem("New palette", None)
            for key, label in PALETTES.items():
                self.combo.addItem(f"Built-in · {label}", ["builtin", key])
            self.combo.insertSeparator(self.combo.count())
            for name in sorted(self.palettes, key=str.casefold):
                self.combo.addItem(name, name)
            selected = ["builtin", self.source_palette] if self.source_palette else self.current
            self.combo.setCurrentIndex(max(0, self.combo.findData(selected)))

    def draft(self) -> dict[str, Any]:
        """
        Validate sorted stop widgets, preserving precise imported positions until edited.
        """
        stops = []
        for row in range(self.stops.rowCount()):
            position = self.stops.cellWidget(row, 0)
            color = self.stops.cellWidget(row, 1)
            assert isinstance(position, QDoubleSpinBox) and isinstance(color, QPushButton)
            stops.append({"position": position.property("position"), "color": color.property("color")})
        stops.sort(key=lambda stop: stop["position"])
        result = validate_palette({"name": self.name.text(), "stops": stops})
        return result

    def load_draft(self, name: str | None) -> None:
        """
        Select a saved palette or initialize a black-to-white draft.
        """
        self.current = name
        self.source_palette = None
        value = self.palettes[name] if name is not None else {
            "name": "Untitled", "stops": [
                {"position": 0, "color": "#000000"}, {"position": 1, "color": "#ffffff"},
            ],
        }
        self.baseline = palette_token(value)
        self.show_draft(value)
        self.refresh_library()

    def show_draft(self, value: dict[str, Any]) -> None:
        """
        Populate bounded stop controls without emitting intermediate draft changes.
        """
        self.building = True
        self.name.setText(value["name"])
        self.stops.setRowCount(len(value["stops"]))
        for row, stop in enumerate(value["stops"]):
            position = QDoubleSpinBox()
            position.setDecimals(1)
            position.setRange(0, 100)
            position.setKeyboardTracking(False)
            position.setValue(stop["position"] * 100)
            position.setProperty("position", stop["position"])
            position.setEnabled(row not in (0, len(value["stops"]) - 1))
            position.installEventFilter(self)
            position.valueChanged.connect(
                lambda amount, field=position: self.move_stop(field, amount)
            )
            color = QPushButton(stop["color"])
            color.setProperty("color", stop["color"])
            color.installEventFilter(self)
            color.clicked.connect(lambda _checked=False, field=color: self.choose_color(field))
            color.setStyleSheet(f"border-bottom: 5px solid {stop['color']}")
            self.stops.setCellWidget(row, 0, position)
            self.stops.setCellWidget(row, 1, color)
        self.building = False
        self.update_preview()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Select the containing stop when interacting with its embedded controls.
        """
        if (event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress)
                and isinstance(watched, (QDoubleSpinBox, QPushButton))):
            index = self.stops.indexAt(watched.pos())
            if index.isValid():
                self.stops.setCurrentCell(index.row(), index.column())
        handled = super().eventFilter(watched, event)
        return handled

    def move_stop(self, field: QDoubleSpinBox, amount: float) -> None:
        """
        Retain numeric keypad support through the standard Qt spin-box editor.
        """
        field.setProperty("position", amount / 100)
        self.update_preview()

    def choose_color(self, button: QPushButton) -> None:
        """
        Edit one RGB stop with a native color picker, honoring cancellation and locks.
        """
        if self.locked:
            return
        color = QColorDialog.getColor(QColor(button.text()), self, "Choose palette stop color")
        if color.isValid() and not self.locked:
            button.setProperty("color", color.name())
            button.setText(color.name())
            button.setStyleSheet(f"border-bottom: 5px solid {color.name()}")
            self.update_preview()

    def update_preview(self) -> None:
        """
        Display the actual interpolated LUT and distinguish saved data from edited drafts.
        """
        if self.building:
            return
        try:
            token = palette_token(self.draft())
            ramp = custom_lut(token)[:, ::-1].copy()
            image = QImage(ramp.data, 256, 1, 768, QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(image))
            self.status.setText("Saved" if self.current and token == self.baseline else "Unsaved draft")
        except ValueError as exc:
            self.status.setText(str(exc))

    def discard_draft(self) -> bool:
        """
        Protect changed drafts before switching palettes, importing or starting over.
        """
        if self.locked:
            return False
        try:
            dirty = palette_token(self.draft()) != self.baseline
        except ValueError:
            dirty = True
        if not dirty:
            return True
        confirmed = QMessageBox.question(
            self, "Discard palette draft", "Discard the unsaved palette changes?",
        ) == QMessageBox.StandardButton.Yes
        return confirmed and not self.locked

    def select(self, index: int) -> None:
        """
        Switch saved palettes only after accounting for unsaved edits.
        """
        if not self.locked and self.discard_draft():
            choice = self.combo.itemData(index)
            if isinstance(choice, list) and len(choice) == 2 and choice[0] == "builtin":
                self.load_builtin(choice[1])
            elif choice is None or isinstance(choice, str):
                self.load_draft(choice)
            else:
                self.refresh_library()
        else:
            self.refresh_library()

    def load_builtin(self, key: str) -> None:
        """
        Copy a built-in node palette into an editable draft without replacing the original.
        """
        if self.locked:
            return
        value = builtin_template(key)
        self.current = None
        self.source_palette = key
        self.baseline = palette_token(value)
        self.show_draft(value)
        self.refresh_library()

    def new(self) -> None:
        """
        Start a fresh palette without removing any saved gradients.
        """
        if not self.locked and self.discard_draft():
            self.load_draft(None)

    def add_stop(self) -> None:
        """
        Split the widest interval, retaining the current interpolated color at its midpoint.
        """
        if self.locked:
            return
        try:
            value = self.draft()
            if len(value["stops"]) >= MAX_STOPS:
                raise ValueError(f"At most {MAX_STOPS} color stops are supported")
            left, right = max(zip(value["stops"], value["stops"][1:]),
                              key=lambda pair: pair[1]["position"] - pair[0]["position"])
            position = round((left["position"] + right["position"]) / 2, 3)
            if not left["position"] < position < right["position"]:
                raise ValueError("Increase the stop spacing before adding another stop")
            color = custom_lut(palette_token(value))[round(position * 255)]
            rgb = "#" + "".join(f"{channel:02x}" for channel in color[::-1])
            value["stops"].append({"position": position, "color": rgb})
            value["stops"].sort(key=lambda stop: stop["position"])
            self.show_draft(value)
        except ValueError as exc:
            self.status.setText(str(exc))

    def remove_stop(self) -> None:
        """
        Remove a selected interior stop while keeping mandatory gradient endpoints.
        """
        row = self.stops.currentRow()
        if not self.locked and 0 < row < self.stops.rowCount() - 1:
            self.stops.removeRow(row)
            self.update_preview()

    def save(self) -> None:
        """
        Save the library atomically and report pipeline validation failures separately.
        """
        if self.locked:
            return
        try:
            self.commit_pending_inputs()
            value = self.draft()
            name = value["name"]
            collision = next((key for key in self.palettes
                              if key.casefold() == name.casefold() and key != self.current), None)
            if collision is not None and QMessageBox.question(
                self, "Replace palette", f'Replace the saved palette "{collision}"?',
            ) != QMessageBox.StandardButton.Yes:
                return
            if self.locked:
                return
            old = palette_token(self.palettes[self.current]) if self.current else None
            candidate = {key: entry for key, entry in self.palettes.items()
                         if key not in (self.current, collision)}
            candidate[name] = value
            save_palettes(candidate)
            self.palettes = candidate
            self.load_draft(name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save palette", str(exc))
            return
        try:
            self.changed(old, palette_token(value))
        except ValueError as exc:
            self.changed(None, None)
            QMessageBox.warning(
                self, "Palette saved; pipeline unchanged",
                f"The palette was saved, but could not be applied to the active pipeline:\n{exc}",
            )

    def commit_pending_inputs(self) -> None:
        """
        Include numeric text drafts in explicit saves and exports.
        """
        if not self.locked:
            for field in self.stops.findChildren(QDoubleSpinBox):
                field.interpretText()

    def delete(self) -> None:
        """
        Delete the library entry while retaining embedded copies in saved or active pipelines.
        """
        if self.locked or self.current is None:
            return
        if QMessageBox.question(
            self, "Delete palette", f'Delete "{self.current}" from the library?',
        ) != QMessageBox.StandardButton.Yes:
            return
        if self.locked:
            return
        try:
            candidate = {name: value for name, value in self.palettes.items() if name != self.current}
            save_palettes(candidate)
            self.palettes = candidate
            self.load_draft(None)
            self.changed(None, None)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not delete palette", str(exc))

    def set_locked(self, locked: bool) -> None:
        """
        Disable library editing during operations that lock pipeline configuration.
        """
        self.locked = locked
        self.content.setEnabled(not locked)
