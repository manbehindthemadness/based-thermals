"""
Keep camera cleanup visible and release the viewer only after the parent disconnects.
"""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from support.qt_process import run_popup

from based_thermals.desktop_app.shutdown import shutdown_stage


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_shutdown_timings_require_debug_logging(
    caplog: pytest.LogCaptureFixture, level: int,
) -> None:
    """
    Suppress timing chatter during normal logging while retaining diagnostics and errors.
    """
    logger = logging.getLogger("based_thermals.test_shutdown")
    diagnostics = Mock()
    caplog.set_level(level, logger=logger.name)
    with (
        pytest.raises(OSError, match="disconnected"),
        shutdown_stage(logger, diagnostics, "camera"),
    ):
        logger.error("Camera disconnected")
        raise OSError("disconnected")
    diagnostics.stage.assert_called_once_with("shutdown/camera")
    assert "Camera disconnected" in caplog.messages
    timings = [record for record in caplog.records if record.message.startswith("Shutdown:")]
    assert len(timings) == (2 if level == logging.DEBUG else 0)
    assert all(record.levelno == logging.DEBUG for record in timings)


def test_close_request_keeps_viewer_visible_until_parent_cleanup(tmp_path: Path) -> None:
    """
    Ignore repeated close requests, freeze controls and close normally after parent EOF.
    """
    script = '''
from PySide6.QtWidgets import QApplication
from based_thermals.viewer_window import ViewerWindow
app = QApplication([])
messages = []
window = ViewerWindow(messages.append)
window.show()
app.processEvents()
assert window.close() is False
assert window.isVisible()
assert [message for message in messages if message.get("action") == "quit"] == [
    {"action": "quit"}
]
assert "Keep the camera connected" in window.status_text.text()
assert not window.toolbar.isEnabled() and not window.canvas.isEnabled()
window.close()
window._command({"action": "capture"})
window.update_state({"unit": "C", "status": "Ready"})
assert "Keep the camera connected" in window.status_text.text()
assert len([message for message in messages if message.get("action") == "quit"]) == 1
assert not any(message.get("action") == "capture" for message in messages)
window.setProperty("parent_disconnected", True)
assert window.close() is True
assert not window.isVisible()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
