"""
Searchable variables, formula editing and native table cells for annotation text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction, QContextMenuEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..markup.expressions import evaluate
from ..markup.tables import cells

Catalog = Callable[[], list[dict[str, Any]]]


def dialog_buttons(parent: QWidget) -> QDialogButtonBox:
    """
    Create explicit OK/Cancel buttons without coupling dialog state to enum flag inference.
    """
    box = QDialogButtonBox(parent)
    box.addButton(QDialogButtonBox.StandardButton.Ok)
    box.addButton(QDialogButtonBox.StandardButton.Cancel)
    return box


def bind(entry: dict[str, Any], bindings: dict[str, str]) -> tuple[str, dict[str, str]]:
    """
    Preserve canonical identities behind readable, collision-free names inserted into formulas.
    """
    for alias, key in bindings.items():
        if key == entry["key"]:
            return alias, bindings
    name = str(entry["alias"]).replace("[", "(").replace("]", ")").replace("\n", " ").replace("\r", " ")[:150]
    alias, index = name, 2
    while alias in bindings:
        alias, index = f"{name} ({index})", index + 1
    return alias, {**bindings, alias: entry["key"]}


def display(value: Any) -> str:
    """
    Show primitive snapshot values concisely without replacing unavailable values with zero.
    """
    return "--" if value is None else f"{value:.6g}" if type(value) is float else str(value)[:100]


class VariablePicker(QDialog):
    """
    Search current system, measurement and pipeline JSON values without memorizing paths.
    """

    def __init__(self, entries: list[dict[str, Any]], parent: QWidget) -> None:
        """
        Present grouped variables with human labels, snapshot values and explicit units.
        """
        super().__init__(parent)
        self.setWindowTitle("Add variable")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search names, measurements, nodes or JSON settings…")
        self.search.setAccessibleName("Find markup variable")
        layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Variable", "Current value", "Unit"])
        self.tree.setColumnWidth(0, 420)
        self.tree.setColumnWidth(1, 180)
        self.groups: dict[str, QTreeWidgetItem] = {}
        for entry in entries:
            group = self.groups.get(entry["group"])
            if group is None:
                group = QTreeWidgetItem(self.tree, [entry["group"]])
                self.groups[entry["group"]] = group
            item = QTreeWidgetItem(group, [entry["label"], display(entry["value"]), entry["unit"]])
            item.setData(0, Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(0, f"[{entry['alias']}]\n{entry['key']}")
        self.tree.expandAll()
        layout.addWidget(self.tree)
        self.description = QLabel("Choose a variable. -- means unavailable in the current snapshot.")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.buttons = dialog_buttons(self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.tree.currentItemChanged.connect(self.selection_changed)
        self.tree.itemDoubleClicked.connect(lambda _item, _column: self.accept() if self.choice() else None)
        self.search.textChanged.connect(self.filter)
        self.selection_changed()

    def choice(self) -> dict[str, Any] | None:
        """
        Return a leaf definition; category headings cannot be inserted as variables.
        """
        item = self.tree.currentItem()
        value = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        return value if isinstance(value, dict) else None

    def selection_changed(self, *_args: Any) -> None:
        """
        Explain the selected human reference while retaining its stable saved identity.
        """
        entry = self.choice()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(entry is not None)
        if entry:
            self.description.setText(f"Insert [{entry['alias']}]\nSaved reference: {entry['key']}")

    def filter(self, text: str) -> None:
        """
        Filter the existing tree in place, leaving values and selection untouched.
        """
        terms = text.casefold().split()
        for name, group in self.groups.items():
            visible = False
            for index in range(group.childCount()):
                item = group.child(index)
                assert item is not None
                entry = item.data(0, Qt.ItemDataRole.UserRole)
                haystack = f"{name} {entry['label']} {entry['alias']} {entry['key']}".casefold()
                match = all(term in haystack for term in terms)
                item.setHidden(not match)
                visible |= match
            group.setHidden(not visible)


class FormulaDialog(QDialog):
    """
    Compose numeric expressions with variable lookup and immediate snapshot validation.
    """

    def __init__(self, entries: Catalog, bindings: dict[str, str], parent: QWidget) -> None:
        """
        Edit a formula without changing the pipeline until the dialog is accepted.
        """
        super().__init__(parent)
        self.setWindowTitle("Add formula")
        self.resize(680, 320)
        self.entries, self.bindings = entries, dict(bindings)
        layout = QVBoxLayout(self)
        hint = QLabel("Use [variables], numbers, + − * / // % ** and parentheses. Functions: abs, min, max, round, sqrt, floor, ceil.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.expression = QPlainTextEdit()
        self.expression.setAccessibleName("Markup formula")
        layout.addWidget(self.expression)
        row = QHBoxLayout()
        add = QPushButton("Add variable…")
        add.clicked.connect(self.add_variable)
        row.addWidget(add)
        self.formatting = QComboBox()
        for label, value in (("Automatic", ""), ("1 decimal", ".1f"), ("2 decimals", ".2f"),
                             ("No decimals", ".0f"), ("Integer", "d"), ("Percentage", ".0%")):
            self.formatting.addItem(label, value)
        row.addWidget(self.formatting)
        layout.addLayout(row)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        self.buttons = dialog_buttons(self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.expression.textChanged.connect(self.validate)
        self.formatting.currentIndexChanged.connect(self.validate)
        self.validate()

    def source(self) -> str:
        """
        Return the expression plus its explicit numeric format, ready for template insertion.
        """
        suffix = self.formatting.currentData()
        return self.expression.toPlainText().strip() + (f" :{suffix}" if suffix else "")

    def validate(self, *_args: Any) -> None:
        """
        Preview the current snapshot and keep invalid arithmetic from being accepted accidentally.
        """
        values = {entry["key"]: entry["value"] for entry in self.entries()}
        try:
            result = evaluate(self.source(), values, self.bindings)
            self.preview.setText(f"Preview: {result}")
            valid = True
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError) as exc:
            self.preview.setText(str(exc))
            valid = False
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def add_variable(self) -> None:
        """
        Insert a readable reference at the formula cursor, retaining stable JSON bindings.
        """
        picker = VariablePicker(self.entries(), self)
        if picker.exec() == QDialog.DialogCode.Accepted:
            entry = picker.choice()
            if entry is not None:
                alias, self.bindings = bind(entry, self.bindings)
                self.expression.insertPlainText(f"[{alias}]")


class TableDialog(QDialog):
    """
    Edit bounded table cells, including the same variable and formula insertion menus.
    """

    def __init__(self, text: str, entries: Catalog, bindings: dict[str, str], parent: QWidget) -> None:
        """
        Keep cell edits and new bindings local until the whole table is accepted.
        """
        super().__init__(parent)
        self.setWindowTitle("Markup table")
        self.resize(760, 450)
        self.entries, self.bindings = entries, dict(bindings)
        layout = QVBoxLayout(self)
        dimensions = QHBoxLayout()
        rows = cells(text)
        self.rows, self.columns = QSpinBox(), QSpinBox()
        self.rows.setRange(1, 32)
        self.columns.setRange(1, 16)
        self.rows.setValue(len(rows))
        self.columns.setValue(len(rows[0]))
        for name, field in (("Rows", self.rows), ("Columns", self.columns)):
            dimensions.addWidget(QLabel(name))
            dimensions.addWidget(field)
        layout.addLayout(dimensions)
        self.table = QTableWidget(len(rows), len(rows[0]))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        layout.addWidget(self.table)
        self.rows.valueChanged.connect(self.table.setRowCount)
        self.columns.valueChanged.connect(self.table.setColumnCount)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.context_menu)
        buttons = dialog_buttons(self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def text(self) -> str:
        """
        Serialize table cells as portable tab-separated rows, preserving formula templates.
        """
        rows = []
        for row in range(self.table.rowCount()):
            values = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                values.append(item.text().replace("\n", " ").replace("\t", " ") if item else "")
            rows.append("\t".join(values))
        return "\n".join(rows)

    def context_menu(self, point: QPoint) -> None:
        """
        Open the same insertion menu on the cell under the mouse, never on a different row.
        """
        index = self.table.indexAt(point)
        if not index.isValid():
            return
        row, column = index.row(), index.column()
        self.table.setCurrentCell(row, column)
        self.table.setFocus()
        menu = QMenu(self)
        variable = menu.addAction("Add variable…")
        formula = menu.addAction("Add formula…")
        chosen = menu.exec(self.table.viewport().mapToGlobal(point))
        if chosen == variable:
            self.insert_variable(row, column)
        elif chosen == formula:
            self.insert_formula(row, column)
        menu.deleteLater()

    def append_cell(self, row: int, column: int, text: str) -> None:
        """
        Update exactly the requested cell after the menu and any modal picker have closed.
        """
        item = self.table.item(row, column)
        self.table.setItem(row, column, QTableWidgetItem((item.text() if item else "") + text))

    def insert_variable(self, row: int, column: int) -> None:
        """
        Open variable lookup after the native menu exits, preserving the table's modal lifetime.
        """
        picker = VariablePicker(self.entries(), self)
        if picker.exec() == QDialog.DialogCode.Accepted:
            entry = picker.choice()
            if entry is not None:
                alias, self.bindings = bind(entry, self.bindings)
                self.append_cell(row, column, "{{ [" + alias + "] }}")

    def insert_formula(self, row: int, column: int) -> None:
        """
        Insert a formula into a cell without accepting the surrounding table prematurely.
        """
        dialog = FormulaDialog(self.entries, self.bindings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.bindings = dialog.bindings
            self.append_cell(row, column, "{{ " + dialog.source() + " }}")


class ReadoutText(QPlainTextEdit):
    """
    Keep normal text editing and add variable/formula/table actions to its context menu.
    """

    def __init__(self, entries: Catalog, bindings: Callable[[], dict[str, str]],
                 save_bindings: Callable[[dict[str, str]], bool], edit_table: Callable[[], None] | None,
                 parent: QWidget | None = None) -> None:
        """
        Use callbacks so the editor never accesses hardware or overwrites live field focus.
        """
        super().__init__(parent)
        self.entries, self.bindings, self.save_bindings, self.edit_table = entries, bindings, save_bindings, edit_table

    def insertion_menu(self) -> QMenu:
        """
        Append discoverable readout actions to the standard Cut/Copy/Paste menu.
        """
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        for title, key, callback in (("Add variable…", "variable", self.add_variable),
                                      ("Add formula…", "formula", self.add_formula),
                                      ("Edit as table…", "table", self.edit_table)):
            if callback is not None:
                action = QAction(title, menu)
                action.setData(key)
                menu.addAction(action)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """
        Preserve the text cursor and selection when opening the requested right-click interface.
        """
        menu = self.insertion_menu()
        chosen = menu.exec(event.globalPos())
        key = chosen.data() if chosen is not None else None
        menu.deleteLater()
        if key == "variable":
            self.add_variable()
        elif key == "formula":
            self.add_formula()
        elif key == "table" and self.edit_table is not None:
            self.edit_table()

    def add_variable(self) -> None:
        """
        Insert one selected readout at the current text cursor, using a friendly saved alias.
        """
        picker = VariablePicker(self.entries(), self)
        if picker.exec() == QDialog.DialogCode.Accepted:
            entry = picker.choice()
            if entry is not None:
                alias, bindings = bind(entry, self.bindings())
                if self.save_bindings(bindings):
                    self.insertPlainText("{{ [" + alias + "] }}")

    def add_formula(self) -> None:
        """
        Insert a validated multi-variable expression without executing Python code.
        """
        dialog = FormulaDialog(self.entries, self.bindings(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self.save_bindings(dialog.bindings):
            self.insertPlainText("{{ " + dialog.source() + " }}")
