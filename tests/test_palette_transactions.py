"""
Palette updates preserve valid pipelines and honor locks acquired during modal dialogs.
"""

from pathlib import Path

import pytest
from support.qt_process import run_popup


def test_palette_replacement_validates_all_branches_before_commit(tmp_path: Path) -> None:
    """
    Reject an oversized Custom config without partially changing App colors or accepted state.
    """
    script = r'''
import json
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QMessageBox
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import node, validate_pipeline
from based_thermals.palettes import palette_token, save_palettes, load_palettes
from based_thermals.pipeline_ui.palette_transfer import refresh_pipeline_palettes
app = QApplication([])
original = {'name':'Gradient', 'stops':[{'position':0, 'color':'#000000'}, {'position':1, 'color':'#ffffff'}]}
replacement = {'name':'Gradient', 'stops':[{'position':i/255, 'color':'#ff0000' if i%2 else '#0000ff'} for i in range(256)]}
old, new = palette_token(original), palette_token(replacement)
save_palettes({'Gradient':original})
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
colors = node('software', 'colors', palette=old)
editor.document['software'].insert(1, colors)
custom = node('software', 'custom',
              config=json.dumps({'ramp':old, 'padding':'x'*65000}),
              controls=json.dumps([{'key':'ramp','label':'Colors','type':'palette'}]))
editor.document['branches']['B'] = [node('software', 'source'), custom]
validate_pipeline(editor.document)
editor.accepted_document = deepcopy(editor.document)
editor.rebuild()
before = deepcopy(editor.document)
serial = editor.edit_serial
try:
    refresh_pipeline_palettes(editor, old, new)
except ValueError as exc:
    assert 'size limit' in str(exc)
else:
    raise AssertionError('Oversized replacement should have been rejected')
assert editor.document == before and editor.accepted_document == before
assert editor.edit_serial == serial and not messages
assert colors['params']['palette'] == old
assert json.loads(custom['params']['config'])['ramp'] == old
validate_pipeline(editor.document)
# Saving the library entry may succeed independently, with an accurate application warning.
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
builder = window.palette_builder
builder.load_draft('Gradient')
builder.show_draft(replacement)
builder.save()
assert load_palettes()['Gradient'] == replacement
assert warnings and warnings[-1][1] == 'Palette saved; pipeline unchanged'
assert editor.document == before and editor.accepted_document == before
assert editor.edit_serial == serial and not messages
validate_pipeline(editor.document)
# A fitting update still commits every active selection and publishes once.
custom['params']['config'] = json.dumps({'ramp':old})
refresh_pipeline_palettes(editor, old, new)
assert colors['params']['palette'] == new
assert json.loads(custom['params']['config'])['ramp'] == new
assert len(messages) == 1 and editor.accepted_document == editor.document
assert messages[0]['hardware_operation'] is False
validate_pipeline(editor.document)
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('action', ['save', 'delete', 'import_file', 'import_collision',
                                    'export', 'color', 'new', 'select'])
def test_palette_dialog_rechecks_locks(tmp_path: Path, action: str) -> None:
    """
    Acquire the lock during a modal response and verify no post-dialog write or draft mutation.
    """
    script = r'''
import sys
from pathlib import Path
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QColorDialog
from based_thermals.pipeline_ui.palette_builder import PaletteBuilder
from based_thermals.pipeline_ui.palette_transfer import import_palettes, export_palette
from based_thermals.palettes import load_palettes, save_palettes, write_library
app = QApplication([])
original = {'name':'Saved', 'stops':[{'position':0, 'color':'#000000'}, {'position':1, 'color':'#ffffff'}]}
save_palettes({'Saved':original})
changes = []
builder = PaletteBuilder(lambda *args: changes.append(args))
builder.load_draft('Saved')
source = Path(sys.argv[1]) / 'incoming.json'
write_library(source, {'Saved':{**original, 'stops':[{'position':0, 'color':'#ff0000'}, {'position':1, 'color':'#ffffff'}]}})
destination = Path(sys.argv[1]) / 'exported.json'
def lock_question(*_args: object) -> QMessageBox.StandardButton:
    """
    Simulate a viewer settings lock arriving while confirmation is open.
    """
    builder.set_locked(True)
    return QMessageBox.StandardButton.Yes
QMessageBox.question = lock_question
if ACTION == 'save':
    builder.load_draft(None)
    builder.name.setText('Saved')
elif ACTION in ('new', 'select'):
    builder.name.setText('Unsaved edit')
before = builder.draft()
current = builder.current
if ACTION == 'save':
    builder.save()
elif ACTION == 'delete':
    builder.delete()
elif ACTION in ('import_file', 'import_collision'):
    def open_file(*_args: object) -> tuple[str, str]:
        """
        Return an import file after an optional settings-lock transition.
        """
        if ACTION == 'import_file':
            builder.set_locked(True)
        return str(source), ''
    QFileDialog.getOpenFileName = open_file
    import_palettes(builder)
elif ACTION == 'export':
    def save_file(*_args: object) -> tuple[str, str]:
        """
        Lock before returning the export destination.
        """
        builder.set_locked(True)
        return str(destination), ''
    QFileDialog.getSaveFileName = save_file
    export_palette(builder)
elif ACTION == 'color':
    def choose_color(*_args: object) -> QColor:
        """
        Lock before returning a newly selected stop color.
        """
        builder.set_locked(True)
        return QColor('#ff0000')
    QColorDialog.getColor = choose_color
    builder.choose_color(builder.stops.cellWidget(0, 1))
elif ACTION == 'new':
    builder.new()
elif ACTION == 'select':
    index = builder.combo.findData(['builtin', 'turbo'])
    builder.combo.setCurrentIndex(index)
    builder.select(index)
assert builder.locked
assert load_palettes() == {'Saved':original}
assert builder.draft() == before and builder.current == current
assert not changes and not destination.exists()
'''
    result = run_popup('ACTION = ' + repr(action) + '\n' + script, tmp_path)
    assert result.returncode == 0, result.stderr
