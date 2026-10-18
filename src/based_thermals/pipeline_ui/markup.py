"""
Portable Markup object controls, with direct drawing delegated to the live viewer.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPoint, QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..markup.clipboard import encode, paste
from ..markup.model import KINDS, TITLES, annotation, read_annotations, serialize
from ..markup.variables import catalog
from ..markup_ui.actions import ObjectActions
from ..pipeline import software_tabs
from .custom_choices import ColorPicker
from .markup_text import ReadoutText, TableDialog

if TYPE_CHECKING:
    from .editor import PipelineEditor


class MarkupControls(QWidget):
    """
    Keep selection/focus local while styles and geometry persist in the node's JSON.
    """

    def __init__(self, editor: PipelineEditor, item: dict[str, Any]) -> None:
        """
        Build object management, drawing, locking and measurement-style appearance fields.
        """
        super().__init__()
        self.editor, self.item = editor, item
        self.updating = False
        self.seen_selection: dict[str, Any] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Display annotations only. Connect this branch to the viewer, then draw on the image.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.lock = QCheckBox("Lock")
        self.lock.setAccessibleName("Lock markup positions")
        self.lock.setToolTip("Prevent on-image drawing and dragging. Styles remain editable.")
        self.lock.toggled.connect(self.set_locked)
        layout.addWidget(self.lock)
        add_line = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.setAccessibleName("Markup drawing tool")
        for kind, title in zip(KINDS, TITLES, strict=True):
            self.kind.addItem(title, kind)
        self.add = QPushButton("Add and draw")
        self.add.clicked.connect(self.add_object)
        add_line.addWidget(self.kind, 1)
        add_line.addWidget(self.add)
        layout.addLayout(add_line)
        self.objects = QListWidget()
        self.objects.setAccessibleName("Markup annotations")
        self.objects.setMinimumHeight(120)
        self.objects.setMaximumHeight(200)
        self.objects.currentRowChanged.connect(self.select_object)
        self.objects.itemClicked.connect(lambda _item: self.select_object(self.objects.currentRow()))
        self.object_actions = ObjectActions(
            self.objects, self.object_action,
            lambda: (self.selected_id() is not None,
                     not self.editor.locked and not self.item["params"]["locked"] and not self.item["bypass"]),
        )
        self.objects.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.objects.customContextMenuRequested.connect(self.object_menu)
        self.objects.itemDoubleClicked.connect(lambda _item: self.text.setFocus())
        layout.addWidget(self.objects)
        actions = QHBoxLayout()
        self.draw = QPushButton("Draw on image")
        self.draw.clicked.connect(self.draw_object)
        self.remove = QPushButton("Delete")
        self.remove.clicked.connect(self.delete_object)
        for button in (self.draw, self.remove):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.fields = QWidget()
        form = QFormLayout(self.fields)
        form.setContentsMargins(0, 0, 0, 0)
        self.z_index = QSpinBox()
        self.z_index.setRange(-1000, 1000)
        self.z_index.setKeyboardTracking(False)
        self.z_index.setAccessibleName("Markup Z index")
        self.z_index.setToolTip("Higher values draw above lower values. Equal values keep their order.")
        self.z_index.valueChanged.connect(partial(self.change, "z_index"))
        self.bring_front = QPushButton("Bring to front")
        self.send_back = QPushButton("Send to back")
        self.bring_front.clicked.connect(lambda: self.shift_z_index(front=True))
        self.send_back.clicked.connect(lambda: self.shift_z_index(front=False))
        z_controls = QHBoxLayout()
        for control in (self.z_index, self.bring_front, self.send_back):
            z_controls.addWidget(control)
        form.addRow("Z index", z_controls)
        self.layer_status = QLabel()
        self.layer_status.setWordWrap(True)
        form.addRow(self.layer_status)
        self.text = ReadoutText(self.variable_catalog, self.variable_bindings, self.save_bindings,
                                self.edit_table)
        self.text.setAccessibleName("Markup text")
        self.text.setMinimumHeight(110)
        self.text.setMaximumHeight(180)
        self.text.setToolTip("Right-click to add a variable, formula or table. Use {{ [Variable] :.1f }} for a formatted readout.")
        self.text.textChanged.connect(lambda: self.change("text", self.text.toPlainText()[:8192]))
        text_hint = QLabel("Text · right-click to add variables, formulas or tables")
        text_hint.setWordWrap(True)
        form.addRow(text_hint)
        form.addRow(self.text)
        self.colors: dict[str, ColorPicker] = {}
        self.numbers: dict[str, QSpinBox | QDoubleSpinBox] = {}
        for role in ("border", "text", "fill"):
            picker = ColorPicker("dynamic", partial(self.change, f"{role}_color"))
            picker.setAccessibleName(f"Markup {role} color")
            self.colors[role] = picker
            form.addRow(f"{role.title()} color", picker)
            opacity = QDoubleSpinBox()
            opacity.setRange(0, 100)
            opacity.setSuffix(" %")
            opacity.setSingleStep(5)
            opacity.setAccessibleName(f"Markup {role} opacity")
            opacity.valueChanged.connect(
                lambda value, opacity_key=f"{role}_opacity": self.change(opacity_key, value / 100)
            )
            self.numbers[f"{role}_opacity"] = opacity
            form.addRow(f"{role.title()} opacity", opacity)
        for key, title, low, high in (("width", "Border width", 1, 12), ("font_size", "Text size", 8, 96)):
            field = QSpinBox()
            field.setRange(low, high)
            field.setSuffix(" px")
            field.setAccessibleName(f"Markup {title.lower()}")
            field.valueChanged.connect(partial(self.change, key))
            self.numbers[key] = field
            form.addRow(title, field)
        self.fill = QCheckBox("Fill shape / text box")
        self.fill.toggled.connect(partial(self.change, "fill"))
        form.addRow(self.fill)
        layout.addWidget(self.fields)
        self.refresh()

    def items(self) -> list[dict[str, Any]]:
        """
        Copy cached validated definitions before applying an explicit UI edit.
        """
        return deepcopy(list(read_annotations(self.item["params"]["annotations"])))

    def selected_id(self) -> str | None:
        """
        Resolve the visible list selection by stable annotation identity.
        """
        selected = self.objects.currentItem()
        return selected.data(Qt.ItemDataRole.UserRole) if selected is not None else None

    def index_of(self, identity: str | None) -> int:
        """
        Locate a saved annotation without depending on its editable label text.
        """
        return next((index for index in range(self.objects.count())
                     if self.objects.item(index).data(Qt.ItemDataRole.UserRole) == identity), -1)

    def select_object(self, _row: int) -> None:
        """
        Synchronize explicit list selection with viewer handles while preserving field focus.
        """
        if self.updating:
            return
        self.seen_selection = self.editor.last_state.get("markup_selection")
        self.editor.send({"action": "markup_select", "node": self.item["id"],
                          "annotation": self.selected_id()})
        self.refresh()

    def save(self, items: list[dict[str, Any]]) -> bool:
        """
        Publish one validated node change through the existing pipeline JSON boundary.
        """
        try:
            encoded = serialize(items)
        except ValueError as exc:
            QMessageBox.warning(self, "Markup", str(exc))
            return False
        self.item["params"]["annotations"] = encoded
        self.editor.publish()
        self.refresh()
        return True

    def variable_catalog(self) -> list[dict[str, Any]]:
        """
        Discover current JSON settings and the latest read-only values published by the viewer.
        """
        state = self.editor.last_state
        return catalog(self.editor.document, state.get("markup_live", {}), state.get("markup_measurements", []))

    def variable_bindings(self) -> dict[str, str]:
        """
        Retrieve only the selected annotation's saved human-name to stable-reference mapping.
        """
        item = next((item for item in self.items() if item["id"] == self.selected_id()), None)
        return {str(key): str(value) for key, value in item["bindings"].items()} if item else {}

    def save_bindings(self, bindings: dict[str, str]) -> bool:
        """
        Publish variable bindings before inserting text, honoring the existing pipeline locks.
        """
        if self.editor.locked:
            return False
        items = self.items()
        selected = next((item for item in items if item["id"] == self.selected_id()), None)
        if selected is None:
            return False
        selected["bindings"] = bindings
        return self.save(items)

    def edit_table(self) -> None:
        """
        Edit cells with variable/formula lookup and preserve all geometry and style on cancellation.
        """
        if self.editor.locked:
            return
        items = self.items()
        selected = next((item for item in items if item["id"] == self.selected_id()), None)
        if selected is None:
            return
        try:
            dialog = TableDialog(self.text.toPlainText(), self.variable_catalog, self.variable_bindings(), self)
        except ValueError as exc:
            QMessageBox.warning(self, "Markup", str(exc))
            return
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        selected.update(kind="table", text=dialog.text(), bindings=dialog.bindings)
        if len(selected["points"]) == 1:
            x, y = selected["points"][0]
            selected["points"].append([min(1., x + .5), min(1., y + .25)])
        if self.save(items):
            self.text.setPlainText(selected["text"])

    def set_locked(self, enabled: bool) -> None:
        """
        Protect canvas positions independently of the global pipeline edit lock.
        """
        if not self.updating and not self.editor.locked:
            self.item["params"]["locked"] = enabled
            self.editor.publish()
            self.refresh()

    def add_object(self) -> None:
        """
        Create an unplaced object and arm its tool in the main viewer.
        """
        if self.editor.locked or self.item["params"]["locked"] or self.item["bypass"]:
            return
        items = self.items()
        if len(items) >= 128:
            return
        item = annotation(self.kind.currentData())
        items.append(item)
        if not self.save(items):
            return
        self.objects.setCurrentRow(self.index_of(item["id"]))
        self.draw_object()

    def draw_object(self) -> None:
        """
        Reposition/redraw the selected object using a live canvas gesture.
        """
        if (not self.editor.locked and not self.item["params"]["locked"]
                and not self.item["bypass"] and self.selected_id()):
            self.editor.send({"action": "markup_draw", "node": self.item["id"],
                              "annotation": self.selected_id()})

    def delete_object(self) -> None:
        """
        Remove only the selected annotation, retaining all other node state.
        """
        if not self.editor.locked and not self.item["params"]["locked"] and not self.item["bypass"]:
            self.save([item for item in self.items() if item["id"] != self.selected_id()])

    def object_menu(self, position: QPoint) -> None:
        """
        Select the right-clicked row before offering object actions in the native menu.
        """
        entry = self.objects.itemAt(position)
        if entry is not None:
            self.objects.setCurrentItem(entry)
        self.object_actions.popup(self.objects.viewport().mapToGlobal(position))

    def object_action(self, operation: str, text: str) -> None:
        """
        Share portable objects with the viewer while leaving text-field clipboard behavior native.
        """
        if self.editor.locked or self.item["params"]["locked"] or self.item["bypass"]:
            return
        items = self.items()
        selected = next((item for item in items if item["id"] == self.selected_id()), None)
        if operation == "paste":
            try:
                added = paste(items, text)
            except ValueError as exc:
                QMessageBox.warning(self, "Markup", str(exc))
                return
            if self.save(items):
                self.objects.setCurrentRow(self.index_of(added["id"]))
        elif selected is not None:
            if operation == "copy":
                QApplication.clipboard().setText(encode(selected))
            elif operation == "delete":
                self.delete_object()
            elif operation == "edit":
                self.text.setFocus(Qt.FocusReason.OtherFocusReason)

    def shift_z_index(self, *, front: bool) -> None:
        """
        Match measurement layer controls across saved Markup nodes without reordering objects.
        """
        if self.editor.locked or not self.selected_id():
            return
        self.z_index.interpretText()
        others = [
            item["z_index"] for nodes in software_tabs(self.editor.document).values() for node in nodes
            if node["type"] == "markup"
            for item in read_annotations(node["params"]["annotations"])
            if (node["id"], item["id"]) != (self.item["id"], self.selected_id())
        ]
        if not others:
            return
        current = self.z_index.value()
        target = max(current, max(others) + 1) if front else min(current, min(others) - 1)
        if not self.z_index.minimum() <= target <= self.z_index.maximum():
            self.layer_status.setText("Z index limit reached; adjust the other annotation layers first.")
            return
        self.z_index.setValue(target)
        self.layer_status.clear()

    def change(self, key: str, value: Any) -> None:
        """
        Apply explicit appearance changes without rebuilding active controls.
        """
        if self.updating or self.editor.locked:
            return
        items = self.items()
        for item in items:
            if item["id"] == self.selected_id():
                item[key] = value
        self.save(items)

    def refresh(self) -> None:
        """
        Reconcile saved changes while preserving focused text, selection and open color popups.
        """
        self.updating = True
        items = self.items()
        options = [(f"{TITLES[KINDS.index(item['kind'])]}: {item['text'][:24]}"
                    + (" · not placed" if not item["points"] else ""), item["id"]) for item in items]
        current_options = [(self.objects.item(i).text(), self.objects.item(i).data(Qt.ItemDataRole.UserRole))
                           for i in range(self.objects.count())]
        if options != current_options:
            identity = self.selected_id()
            blocker = QSignalBlocker(self.objects)
            self.objects.clear()
            for title, identity_value in options:
                entry = QListWidgetItem(title, self.objects)
                entry.setData(Qt.ItemDataRole.UserRole, identity_value)
            self.objects.setCurrentRow(max(0, self.index_of(identity)))
            blocker.unblock()
        incoming = self.editor.last_state.get("markup_selection")
        if (incoming != self.seen_selection and isinstance(incoming, dict)
                and incoming.get("node") == self.item["id"]):
            identity = incoming.get("annotation")
            if isinstance(identity, str) and self.selected_id() != identity:
                blocker = QSignalBlocker(self.objects)
                self.objects.setCurrentRow(self.index_of(identity))
                entry = self.objects.currentItem()
                if entry is not None:
                    self.objects.scrollToItem(entry)
                blocker.unblock()
        self.seen_selection = incoming if isinstance(incoming, dict) else None
        selected = next((item for item in items if item["id"] == self.selected_id()), None)
        self.lock.setChecked(self.item["params"]["locked"])
        self.setEnabled(not self.editor.locked and not self.item["bypass"])
        self.fields.setEnabled(selected is not None)
        self.add.setEnabled(not self.item["params"]["locked"] and len(items) < 128)
        self.draw.setEnabled(selected is not None and not self.item["params"]["locked"])
        self.remove.setEnabled(selected is not None and not self.item["params"]["locked"])
        if selected is not None:
            if not self.z_index.hasFocus() and self.z_index.value() != selected["z_index"]:
                self.z_index.setValue(selected["z_index"])
            has_area = selected["kind"] not in ("line", "arrow")
            self.text.setEnabled(has_area)
            self.fill.setEnabled(has_area)
            self.colors["fill"].setEnabled(has_area)
            self.colors["text"].setEnabled(has_area)
            for key in ("fill_opacity", "text_opacity", "font_size"):
                self.numbers[key].setEnabled(has_area)
            if not self.text.hasFocus() and self.text.toPlainText() != selected["text"]:
                self.text.setPlainText(selected["text"])
            self.fill.setChecked(selected["fill"])
            for role, picker in self.colors.items():
                value = selected[f"{role}_color"]
                if picker.combo.currentData() != value and not picker.combo.view().isVisible():
                    picker.refresh(value)
            for key, field in self.numbers.items():
                value = selected[key] * 100 if key.endswith("opacity") else selected[key]
                if not field.hasFocus() and field.value() != value:
                    field.setValue(value)
        self.updating = False
