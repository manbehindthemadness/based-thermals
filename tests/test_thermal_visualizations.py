"""
Exercise portable visualization packages with synthetic display pixels and no camera/GPU.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.frames import make_frame
from support.pipeline import raw_pipeline

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.custom_nodes.visualization import palette_lut, palette_names
from based_thermals.pipeline import node, validate_pipeline
from based_thermals.pipeline_model.catalog import PALETTES
from based_thermals.pipeline_model.measurement_graphics import measurement_graphics_visible
from based_thermals.processing.images import colorize
from based_thermals.processing.scheduler import PipelineProcessor

FOLDERS = Path(__file__).resolve().parents[1] / "examples" / "custom_nodes"
NAMES = (
    "Thermal Contours",
    "Thermal Shaded Relief",
    "Thermal History",
    "Thermal Surface",
    "Thermal Surface Taichi",
)


@pytest.mark.parametrize("name", NAMES)
def test_only_projected_examples_hide_measurement_graphics(name: str) -> None:
    """
    Preserve overlays for aligned images and use literal metadata for projected output.
    """
    item = configured(name)
    document = validate_pipeline(raw_pipeline(item))
    assert measurement_graphics_visible(document) == (name in NAMES[:2])
    document["software"][2]["bypass"] = True
    assert measurement_graphics_visible(document)


def configured(name: str, **values: Any) -> dict[str, Any]:
    """
    Snapshot a real package and merge current values without altering its literal controls.
    """
    item = node("software", "custom", **load_folder(FOLDERS / name))
    config = json.loads(item["params"]["config"])
    config.update(values)
    item["params"]["config"] = json.dumps(config)
    return item


def changed(item: dict[str, Any], **values: Any) -> dict[str, Any]:
    """
    Preserve the node identity and package while exercising saved configuration changes.
    """
    result = deepcopy(item)
    config = json.loads(result["params"]["config"])
    result["params"]["config"] = json.dumps({**config, **values})
    return result


@pytest.fixture
def processor() -> Any:
    """
    Release temporary package namespaces even when a regression assertion fails.
    """
    instance = CustomProcessor(apple_available=False, nvidia_available=False)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("name", NAMES)
def test_all_palettes_render_and_survive_export(name: str, processor: CustomProcessor) -> None:
    """
    Check every visible choice through real package execution and a JSON document round trip.
    """
    item = configured(
        name,
        resolution="384",
        detail=16,
        samples=16,
        history=8,
        execution="OpenCV",
        antialiasing="Off",
    )
    controls = json.loads(item["params"]["controls"])
    palette = next(field for field in controls if field["key"] == "palette")
    assert tuple(palette["options"]) == palette_names()
    image = np.repeat(np.linspace(0, 255, 32, dtype=np.float32)[None, :, None], 3, axis=2)
    image = np.repeat(image, 24, axis=0)
    original = image.copy()
    for palette_name in palette_names():
        selected = changed(item, palette=palette_name)
        document = validate_pipeline(json.loads(json.dumps(raw_pipeline(selected))))
        output = processor.apply(image, document["software"][2])
        assert output.ndim == 3 and output.shape[2] == 3
        assert output.dtype == np.float32 and np.isfinite(output).all()
        assert 0 <= output.min() <= output.max() <= 255
    np.testing.assert_array_equal(image, original)
    assert len(processor.packages) == 1


@pytest.mark.parametrize("key,label", PALETTES.items())
def test_ramps_match_application_colorization(key: str, label: str) -> None:
    """
    Compare all 256 colors with the public pipeline, including inverted grayscale and aliases.
    """
    name = f"{label} (approx.)" if key.startswith("camera_") else label
    expected = colorize(np.arange(256, dtype=np.uint8).reshape(256, 1), key)[:, 0]
    np.testing.assert_array_equal(palette_lut(name), expected)
    assert not palette_lut(name).flags.writeable


@pytest.mark.parametrize("name", NAMES[:3])
def test_raw_pipeline_keeps_measurements_and_ownership(name: str) -> None:
    """
    Run the actual branch scheduler without changing native camera-frame bytes.
    """
    item = configured(name, resolution="384", samples=16, history=8)
    document = validate_pipeline(raw_pipeline(item))
    frame = make_frame()
    original = bytes(frame)
    pipeline = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, source = pipeline.process(frame, None, document, scale=1, rotation=90)
        assert source == "raw" and output.shape == (256, 192, 3)
        assert frame == original
    finally:
        pipeline.close()


def test_contour_bands_and_line_controls(processor: CustomProcessor) -> None:
    """
    Verify independently calculated band centers, editable ink and constant-field behavior.
    """
    item = configured("Thermal Contours", palette="Grayscale", smoothing=0, levels=4, lines=False)
    gray = np.array((0, 64, 128, 192, 255), np.float32)
    image = np.repeat(np.tile(gray, (8, 1))[:, :, None], 3, axis=2)
    output = processor.apply(image, item)
    expected = np.tile(np.array((32, 96, 159, 223, 223), np.float32), (8, 1))
    np.testing.assert_array_equal(output[:, :, 0], expected)
    lines = processor.apply(image, changed(item, lines=True, line_color="#FF0000", line_opacity=1))
    np.testing.assert_array_equal(lines[:, 1:4], np.tile((0, 0, 255), (8, 3, 1)))
    smooth = processor.apply(image, changed(item, bands=False))
    np.testing.assert_array_equal(smooth, image)
    flat = processor.apply(np.full((1, 1, 3), 128, np.float32), changed(item, lines=True))
    np.testing.assert_array_equal(flat, np.full((1, 1, 3), 159))
    middle = np.repeat(np.tile((127, 127, 128, 128, 128), (8, 1))[:, :, None], 3, axis=2)
    contrasted = processor.apply(middle.astype(np.float32), changed(
        item, levels=2, bands=False, lines=True, line_color="dynamic", line_opacity=1,
    ))
    np.testing.assert_array_equal(contrasted[:, 2], np.full((8, 3), 127))
    assert contrasted[4, 1, 0] == 0 and contrasted[4, 3, 0] == 255


def test_relief_has_independent_flat_light_and_opposing_slopes(processor: CustomProcessor) -> None:
    """
    Check a flat plane analytically and reverse the light across a rising horizontal ramp.
    """
    item = configured(
        "Thermal Shaded Relief", palette="Grayscale", smoothing=0, elevation=30, ambient=0.2
    )
    flat = np.full((16, 16, 3), 200, np.float32)
    result = processor.apply(flat, item)
    np.testing.assert_allclose(result, 200 * (0.2 + 0.8 * 0.5), atol=1e-4)
    ramp = np.repeat(
        np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))[:, :, None], 3, axis=2
    )
    right = processor.apply(ramp, changed(item, azimuth=0))
    left = processor.apply(ramp, changed(item, azimuth=180))
    assert np.all(left[:, 1:-1] > right[:, 1:-1])
    unlit = processor.apply(ramp, changed(item, mix=0))
    np.testing.assert_allclose(unlit, ramp, atol=0.5)


def test_history_ring_freeze_cadence_reset_and_node_isolation(processor: CustomProcessor) -> None:
    """
    Keep only the last eight scalar profiles and preserve history during view/palette changes.
    """
    item = configured("Thermal History", history=8, samples=16, resolution="384", smoothing=0)
    for value in range(25):
        processor.apply(np.full((12, 16, 3), value, np.float32), item)
    module = sys.modules[processor.packages[item["id"]].name]
    renderer = module._renderer
    assert renderer.count == 8 and renderer.buffer.nbytes == 8 * 16 * 4
    indices = (np.arange(8) + renderer.head - 8) % 8
    np.testing.assert_array_equal(renderer.buffer[indices, 0], np.arange(17, 25))
    buffer = renderer.buffer.copy()
    frozen = changed(item, freeze=True)
    first = processor.apply(np.full((12, 16, 3), 200, np.float32), frozen)
    second = processor.apply(np.zeros((12, 16, 3), np.float32), frozen)
    np.testing.assert_array_equal(first, second)
    processor.apply(np.zeros((12, 16, 3), np.float32), changed(frozen, yaw=30, palette="Magma"))
    np.testing.assert_array_equal(renderer.buffer, buffer)
    separate = deepcopy(item)
    separate["id"] = "second-history-node"
    processor.apply(np.full((12, 16, 3), 99, np.float32), separate)
    other = sys.modules[processor.packages[separate["id"]].name]._renderer
    assert other.count == 1 and not np.shares_memory(other.buffer, renderer.buffer)
    reset = changed(item, reset=1, every=3)
    for value in range(7):
        processor.apply(np.full((12, 16, 3), value, np.float32), reset)
    assert renderer.count == 3
    np.testing.assert_array_equal(renderer.buffer[:3, 0], (0, 3, 6))
    processor.apply(np.zeros((6, 16, 3), np.float32), changed(reset, freeze=True))
    assert renderer.count == 0


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("Mean rows", np.arange(16) + 15),
        ("Mean columns", 2 * np.arange(16) + 7.5),
        ("Horizontal line", np.arange(16) + 30),
        ("Vertical line", 2 * np.arange(16) + 15),
    ],
)
def test_history_profile_sampling(
    profile: str, expected: np.ndarray, processor: CustomProcessor
) -> None:
    """
    Check mean and end-position line profiles against independently constructed scalar planes.
    """
    plane = np.add.outer(2 * np.arange(16), np.arange(16)).astype(np.float32)
    image = np.repeat(plane[:, :, None], 3, axis=2)
    item = configured(
        "Thermal History",
        profile=profile,
        position=100,
        smoothing=0,
        samples=16,
        history=8,
        resolution="384",
    )
    processor.apply(image, item)
    module = sys.modules[processor.packages[item["id"]].name]
    np.testing.assert_array_equal(module._renderer.buffer[0], expected)


@pytest.mark.parametrize("name", NAMES[:3])
@pytest.mark.parametrize(
    "values",
    [
        {"palette": []},
        {"smoothing": True},
        {"smoothing": float("nan")},
        {"normalization": "Celsius"},
    ],
)
def test_standalone_validation(
    name: str, values: dict[str, Any], processor: CustomProcessor
) -> None:
    """
    Reject malformed values even when callers bypass JSON validation.
    """
    item = configured(name, resolution="384", samples=16, history=8)
    image = np.zeros((2, 2, 3), np.float32)
    processor.apply(image, item)
    callback = processor.packages[item["id"]].process
    config = {**json.loads(item["params"]["config"]), **values}
    with pytest.raises(ValueError):
        callback(image, config)
    bad_image = image.copy()
    bad_image[:, :, 2] = 255
    with pytest.raises(ValueError, match="Source Raw"):
        callback(bad_image, json.loads(item["params"]["config"]))


@pytest.mark.parametrize("name,values", [
    ("Thermal Contours", {"levels": 65}),
    ("Thermal Contours", {"line_width": 0}),
    ("Thermal Contours", {"line_color": "red"}),
    ("Thermal Contours", {"bands": 1}),
    ("Thermal Shaded Relief", {"elevation": 0}),
    ("Thermal Shaded Relief", {"strength": -1}),
    ("Thermal Shaded Relief", {"mix": 1.1}),
    ("Thermal History", {"samples": 257}),
    ("Thermal History", {"samples": 16.5}),
    ("Thermal History", {"history": 0}),
    ("Thermal History", {"every": 0}),
    ("Thermal History", {"reset": -1}),
    ("Thermal History", {"zoom": 0}),
    ("Thermal History", {"resolution": "4096"}),
    ("Thermal History", {"freeze": 1}),
])
def test_invalid_node_settings_fail_before_state_changes(
    name: str, values: dict[str, Any], processor: CustomProcessor,
) -> None:
    """
    Reject unsupported allocations and malformed controls even outside the editor schema.
    """
    item = configured(name, resolution="384", samples=16, history=8)
    image = np.zeros((2, 2, 3), np.float32)
    processor.apply(image, item)
    callback = processor.packages[item["id"]].process
    with pytest.raises((ValueError, TypeError)):
        callback(image, {**json.loads(item["params"]["config"]), **values})
    if name == "Thermal History":
        renderer = sys.modules[processor.packages[item["id"]].name]._renderer
        assert renderer.count == 1


@pytest.mark.parametrize("name", NAMES[:3])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1)])
def test_tiny_flat_and_portrait_inputs(name: str, shape: tuple[int, int],
                                      processor: CustomProcessor) -> None:
    """
    Keep degenerate spatial dimensions and constant auto-contrast fields finite.
    """
    item = configured(name, normalization="Auto contrast", resolution="384", samples=16, history=8)
    output = processor.apply(np.full((*shape, 3), 128, np.float32), item)
    assert np.isfinite(output).all() and output.dtype == np.float32
    assert output.shape == ((288, 384, 3) if name == "Thermal History" else (*shape, 3))


def test_history_maximum_empty_freeze_projection_and_shared_window_range(
    processor: CustomProcessor,
) -> None:
    """
    Bound buffers/output, retain honest empty history and normalize all stored rows together.
    """
    item = configured("Thermal History", samples=256, history=256, resolution="1920",
                      freeze=True, smoothing=0, yaw=0, tilt=45)
    image = np.zeros((12, 16, 3), np.float32)
    empty = processor.apply(image, item)
    renderer = sys.modules[processor.packages[item["id"]].name]._renderer
    assert renderer.count == 0 and renderer.buffer.nbytes == 256 * 1024
    assert empty.shape == (1440, 1920, 3) and empty.max() < 100
    mesh = renderer.mesh
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), np.float32)
    pixels, depth = mesh.project(vertices)
    expected = np.array(((0, 0), (1, 0), (0, -2**-0.5), (0, -2**-0.5)))
    np.testing.assert_allclose((pixels - pixels[0]) / mesh.scale, expected, atol=1e-6)
    np.testing.assert_allclose(depth, (0, 0, 2**-0.5, -2**-0.5), atol=1e-6)
    item = changed(item, samples=16, history=8, resolution="384", freeze=False)
    for value in (50, 150):
        processor.apply(np.full_like(image, value), item)
    frozen = changed(item, freeze=True, normalization="Auto contrast")
    actual = processor.apply(image, frozen)
    expected_heat = np.array(((0,) * 16, (1,) * 16), np.float32)
    expected_image = renderer.mesh.render(expected_heat, palette_lut("Turbo"), False)
    np.testing.assert_array_equal(actual, expected_image)
