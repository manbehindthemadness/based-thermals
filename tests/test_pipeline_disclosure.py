"""
Persist software section presentation across independent Imaging panel processes.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_software_disclosure_survives_panel_restarts(tmp_path: Path) -> None:
    """
    Restore both disclosure states without publishing pipeline or camera changes.
    """
    setup = '''
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
'''
    steps = (
        '''
assert editor.software_expanded
editor.toggle_software_section()
assert not editor.software_expanded
assert editor.stacks['software'].isHidden()
assert not messages
''',
        '''
assert not editor.software_expanded
assert editor.stacks['software'].isHidden()
editor.rebuild()
editor.update_state({}, False)
assert not editor.software_expanded
editor.toggle_software_section()
assert editor.software_expanded
assert not messages
''',
        '''
assert editor.software_expanded
assert not editor.stacks['software'].isHidden()
assert not messages
''',
    )
    for step in steps:
        result = run_popup(setup + step + '\nwindow.close()\n', tmp_path)
        assert result.returncode == 0, result.stderr
