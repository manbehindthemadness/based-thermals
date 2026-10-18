"""
Gate native provider warning diagnostics on the persisted application debug switch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from support.onnx_visual import fake_runtime as fake_visual_runtime
from test_coreml_upsampling import fake_runtime as fake_apple_runtime

# noinspection PyProtectedMember
from based_thermals.coreml_upsampling import _CoreMLRuntime
from based_thermals.maintenance.logging import save_debug
from based_thermals.onnx_upsampling import ONNXRuntime


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "coreml", "apple-acnet"])
def test_native_warning_verbosity_follows_saved_debug_preference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, debug: bool, backend: str,
) -> None:
    """
    Read the saved switch for new sessions even without a configured Python debug logger.
    """
    monkeypatch.setattr("based_thermals.maintenance.logging._path", lambda: tmp_path / "viewer.json")
    save_debug(debug)
    image = np.zeros((4, 6), np.uint8)
    if backend == "apple-acnet":
        calls = fake_apple_runtime(monkeypatch)
        _CoreMLRuntime().apply(image)
    else:
        calls = fake_visual_runtime(monkeypatch)
        monkeypatch.setattr(sys, "platform", "darwin")
        ONNXRuntime().apply(image, "espcn", backend=backend)
    # ORT: WARNING=2 when debugging, ERROR=3 otherwise; errors remain visible.
    assert calls[0]["sess_options"].log_severity_level == (2 if debug else 3)
