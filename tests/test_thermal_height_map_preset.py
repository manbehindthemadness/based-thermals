"""
Ship the saved height-map layout as a portable, immediately usable preset.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from support.frames import make_frame
from test_pipeline_presets import collapsed_nodes

from based_thermals.camera_backends.duo.camera import IMAGE_OFFSET, SENSOR_PIXELS
from based_thermals.custom_nodes.bundle import load_folder, package_files
from based_thermals.pipeline import validate_pipeline
from based_thermals.pipeline_presets import load_presets, predefined_presets
from based_thermals.processing.scheduler import PipelineProcessor

ROOT = Path(__file__).resolve().parents[1]
NAME = "Thermal Height Map"


def test_height_map_is_bundled_with_portable_source_and_saved_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Load without local presets and preserve the captured configuration through JSON export.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    export = json.loads((ROOT / "presets" / f"{NAME}.pipeline.json").read_text())
    document = predefined_presets()[NAME]
    assert document == collapsed_nodes(validate_pipeline(export))
    assert load_presets()[NAME] == document
    assert validate_pipeline(json.loads(json.dumps(document))) == document
    custom = next(item for item in document["software"] if item["type"] == "custom")
    params = custom["params"]
    example = load_folder(ROOT / "examples" / "custom_nodes" / "Thermal Surface")
    assert package_files(params["package"]) == package_files(example["package"])
    assert json.loads(params["config"]) == {
        "detail": 60, "yaw": 80, "tilt": 45, "height": 0.8, "smoothing": 0.6,
        "palette": "Turbo", "normalization": "Source range", "resolution": "960",
        "wireframe": False, "scene_rotation": 90, "offset_x": -14, "offset_y": 0, "zoom": 1,
    }
    assert len(json.loads(params["controls"])) == 13
    assert document["software"][0]["params"]["source"] == "preview"
    assert set(document["branches"]) == {"B", "C"}
    assert document["branches"]["B"][0]["params"]["source"] == "preview"
    assert document["branches"]["C"][0]["params"]["source"] == "raw"
    contrast = next(item for item in document["hardware"] if item["type"] == "contrast")
    assert contrast["params"]["value"] == 25
    markup = next(item for item in document["software"] if item["type"] == "markup")
    assert markup["params"]["locked"] is True
    annotations = json.loads(markup["params"]["annotations"])
    assert [item["kind"] for item in annotations] == [
        "text", "text", "arrow", "arrow", "textbox",
    ]
    assert [item["text"] for item in annotations if item["kind"] == "text"] == [
        "Hardware Upsample", "Raw Thermal Sensor\n",
    ]
    assert annotations[-1]["text"] == "3D height map representing raw thermal sensor output"
    assert all(item["text_color"] == "dynamic" for item in annotations)
    assert not list(tmp_path.rglob("pipeline-presets.json"))


def test_embedded_height_map_renders_with_insets_without_external_resources() -> None:
    """
    Execute the shipped package and connected branches on a synthetic grayscale camera frame.
    """
    words = np.frombuffer(make_frame(4800), dtype="<u2").copy()
    words[IMAGE_OFFSET:IMAGE_OFFSET + SENSOR_PIXELS] = (
        np.arange(SENSOR_PIXELS, dtype=np.uint16) % 256
    ) | 0x8000
    frame = words.tobytes()
    original = bytes(frame)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        image, source = processor.process(
            frame, None, predefined_presets()[NAME], scale=1, rotation=90,
        )
        assert image.shape == (256, 192, 3) and image.dtype == np.uint8
        assert np.isfinite(image).all() and np.ptp(image) > 50
        assert np.any(image[..., 0] != image[..., 2])
        assert source == "raw" and frame == original  # The final combine supplies the Raw inset.
        assert set(processor.branches) == {"A", "B", "C"}
        assert len(processor.branches["A"].custom.packages) == 1
        assert not processor.last_preview_errors
    finally:
        processor.close()
