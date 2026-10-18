"""
Keep hardware nodes aligned with restored camera defaults without changing software branches.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture

from based_thermals import desktop
from based_thermals.desktop_app.restored_nodes import restored_hardware_nodes
from based_thermals.desktop_app.session import DesktopSession
from based_thermals.pipeline import node, validate_pipeline


def test_duo_nodes_use_restored_values_instead_of_catalog_defaults(viewer: Any) -> None:
    """
    Represent each display group, including restored preset/tone state and camera palette.
    """
    hardware = viewer.hardware
    hardware.state.return_value["brightness"]["value"] = 44
    hardware.state.return_value["contrast"]["value"] = 37
    hardware.state.return_value["palette"]["value"] = 12
    hardware.state.return_value["detail"]["value"] = 63
    hardware.state.return_value["detail_enabled"]["value"] = 1
    hardware.processing_preset = "soft"
    result = restored_hardware_nodes(hardware, "duo")
    by_kind = {item["type"]: item for item in result}
    assert len(result) == 8 and len(by_kind) == 8
    assert by_kind["brightness"]["params"] == {"value": 44}
    assert by_kind["contrast"]["params"] == {"value": 37}
    assert by_kind["camera_colors"]["params"] == {"palette": 12}
    assert by_kind["detail"]["params"] == {"amount": 63, "enabled": True, "fixed": False}
    assert by_kind["preset"]["params"] == {"value": "soft"}
    assert by_kind["gamma"]["params"] == {"value": 50}
    assert by_kind["boost"]["params"] == {"value": 0}
    assert all(not item["bypass"] and not item["expanded"] for item in result)
    hardware.set.assert_not_called()
    hardware.set_pipeline.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_restore_button_replaces_hardware_nodes_and_persists_only_after_success(
    viewer: Any, failure: bool,
) -> None:
    """
    Rebuild hardware rows after restore while retaining software, branches and failure behavior.
    """
    session = DesktopSession(desktop, [])
    session.pipeline["hardware"] = [node("hardware", "contrast", value=91)]
    session.pipeline["branches"] = {"B": [node("software", "source", source="raw")]}
    before = deepcopy(session.pipeline)
    session.view_panel = Mock()
    session.view_panel.poll.return_value = [{"action": "restore_hardware"}]
    if failure:
        viewer.hardware.restore.side_effect = desktop.CameraError("restore failed")
    try:
        session.poll_camera_commands()
        viewer.hardware.restore.assert_called_once()
        if failure:
            assert session.pipeline == before
            assert "restore failed" in viewer.hardware.error
            return
        assert session.pipeline["software"] == before["software"]
        assert session.pipeline["branches"] == before["branches"]
        assert len(session.pipeline["hardware"]) == 8
        assert all(not item["bypass"] for item in session.pipeline["hardware"])
        assert validate_pipeline(session.pipeline) == session.pipeline
        assert desktop.load_settings()["pipeline"] == session.pipeline
        assert session.renderer.pipeline == session.pipeline
        assert session.pipeline_revision == 1
        viewer.hardware.set_pipeline.assert_not_called()
        viewer.hardware.set.assert_not_called()
        session.poll_camera_commands()
        assert len(session.pipeline["hardware"]) == 8  # Repeated restores do not duplicate nodes.
    finally:
        session.renderer.pipeline_processor.close()
        session.support_export.close()
        session.support_logging.close()


def test_other_cameras_only_generate_available_device_preview_controls() -> None:
    """
    Keep measurement, unavailable and spot-scoped controls outside the hardware pipeline.
    """
    specs = {
        "brightness": SimpleNamespace(supported=True, scope="device", effect="preview"),
        "ambient": SimpleNamespace(supported=True, scope="device", effect="measurement"),
        "spot": SimpleNamespace(supported=True, scope="spot", effect="preview"),
        "missing": SimpleNamespace(supported=True, scope="device", effect="preview"),
    }
    hardware = SimpleNamespace(specs=specs, state=lambda: {
        "brightness": {"value": 35, "available": True},
        "ambient": {"value": 22, "available": True},
        "spot": {"value": 50, "available": True},
    })
    result = restored_hardware_nodes(hardware, "example")
    assert len(result) == 1
    assert result[0]["type"] == "device_control"
    assert result[0]["params"] == {"control": "brightness", "value": 35}


__all__ = ["viewer_fixture"]
