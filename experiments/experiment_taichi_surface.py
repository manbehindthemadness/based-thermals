"""
Compare isolated optional Taichi backends with OpenCV through real Custom callbacks, without USB.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from experiment_thermal_surface import heat_image

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import default_pipeline, node, validate_pipeline

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "examples/custom_nodes/Thermal Surface Taichi"
EXECUTIONS = {"opencv": "OpenCV", "cpu": "Taichi CPU", "metal": "Taichi Metal",
              "cuda": "Taichi CUDA", "vulkan": "Taichi Vulkan"}


def rss_kib() -> int | None:
    """
    Sample current resident memory outside timed sections, retaining native allocation high-water effects.
    """
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return int(result.stdout.strip()) if result.returncode == 0 else None


def configured(params: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Exercise portable JSON save/import validation before running the resulting node.
    """
    item = node("software", "custom", **{**params, "config": json.dumps(config)})
    document = default_pipeline()
    document["software"].insert(-1, item)
    restored = validate_pipeline(json.loads(json.dumps(document)))
    return next(n for n in restored["software"] if n["id"] == item["id"])


def renderer_state(processor: CustomProcessor, item: dict[str, Any]) -> Any:
    """
    Read the experimental node's own execution evidence without adding application interfaces.
    """
    return vars(sys.modules[processor.packages[item["id"]].name])["_renderer"]


def check_controls(processor: CustomProcessor, params: dict[str, Any],
                   config: dict[str, Any], output: Path) -> None:
    """
    Check source handedness against independent known quadrants and exercise the remaining view controls.
    """
    image = np.zeros((48, 64, 3), np.float32)
    image[6:18, 42:56] = 255
    for rotation, right, bottom in ((0, True, False), (90, True, True),
                                    (180, False, True), (-90, False, False)):
        item = configured(params, {**config, "yaw": 0, "tilt": 85, "height": 0,
                                   "smoothing": 0, "palette": "Grayscale", "scene_rotation": rotation,
                                   "offset_x": 0, "offset_y": 0, "zoom": 1})
        result = processor.apply(image, item)
        hot_pixels = np.argwhere(result[:, :, 0] > 150)
        assert hot_pixels.size > 0
        centroid = hot_pixels.mean(axis=0)
        assert (bool(centroid[1] > result.shape[1] / 2),
                bool(centroid[0] > result.shape[0] / 2)) == (right, bottom)
        assert renderer_state(processor, item).effective_execution == config["execution"]
        processor.retain(set())
    plain_item = configured(params, config)
    plain = processor.apply(image, plain_item)
    for changes in ({"wireframe": True}, {"zoom": 1.7, "offset_x": 15, "offset_y": -10}):
        item = configured(params, {**config, **changes})
        changed = processor.apply(image, item)
        assert changed.shape == plain.shape and not np.array_equal(changed, plain)
        if changes.get("wireframe"):
            cv2.imwrite(str(output / f"{config['execution'].replace(' ', '-')}-wireframe.png"),
                        changed.astype(np.uint8))
    portrait = processor.apply(image.transpose(1, 0, 2).copy(), plain_item)
    assert portrait.shape == (384, 288, 3)
    processor.retain(set())
    if config.get("geometry_method") == "Taichi" and config["execution"] != "OpenCV":
        for normalization in ("Source range", "Auto contrast"):
            for antialiasing in ("Off", "2x supersampling"):
                values = {**config, "normalization": normalization, "antialiasing": antialiasing}
                native = configured(params, values)
                baseline = configured(params, {**values, "geometry_method": "NumPy"})
                actual = processor.apply(image, native)
                expected = processor.apply(image, baseline)
                assert renderer_state(processor, native).effective_execution == config["execution"]
                error: npt.NDArray[np.float32] = np.abs(actual - expected)
                channel_error: npt.NDArray[np.float32] = error.max(axis=2)
                difference_fraction = float(np.count_nonzero(channel_error > 8)) / channel_error.size
                assert error.mean() < 1 and difference_fraction < .01
                processor.retain(set())


def child(args: argparse.Namespace) -> dict[str, Any]:
    """
    Measure one backend per disposable process; completed GPU readback is part of every frame.
    """
    params = load_folder(FOLDER)
    reference = load_folder(ROOT / "examples/custom_nodes/Thermal Surface")
    defaults = json.loads(params["config"])
    image = heat_image()
    original = image.copy()
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    rows = []
    try:
        cases = ([(detail, "960") for detail in args.details] if args.details
                 else [(24, "384"), (40, "640"), (64, "960"), (96, "960")])
        for detail, resolution in cases:
            resolution = args.resolution or resolution
            config = {**defaults, "detail": detail, "resolution": resolution,
                      "antialiasing": args.antialiasing,
                      "geometry_method": args.geometry,
                      "execution": EXECUTIONS[args.child]}
            item = configured(params, config)
            original_reference = (detail <= 96 and resolution in ("384", "640", "960")
                                  and args.antialiasing == "Off")
            reference_params = reference if original_reference else params
            reference_item = configured(reference_params, {**config, "execution": "OpenCV",
                                                            "geometry_method": "NumPy"})
            start = perf_counter()
            result = processor.apply(image, item)
            cold_ms = (perf_counter() - start) * 1000
            renderer = renderer_state(processor, item)
            if renderer.effective_execution != config["execution"]:
                raise RuntimeError(f"Requested {config['execution']}; fell back: {renderer.fallback_reason}")
            if args.geometry == "Taichi" and args.child != "opencv":
                mesh = renderer.rasterizer.mesh
                assert mesh is not None
                vertices = mesh.vertices.to_numpy()
                pixels, depths = renderer.geometry.project(vertices)
                np.testing.assert_allclose(mesh.pixels.to_numpy(), pixels, rtol=1e-5, atol=.002)
                np.testing.assert_allclose(mesh.depths.to_numpy(), depths, rtol=1e-5, atol=1e-6)
            if args.antialiasing != "Off" and args.child != "opencv":
                floor = np.zeros((16, 16, 3), np.uint8)
                triangles = np.array([[[1.25, 1.25], [8.25, 1.25], [1.25, 8.25]]], np.float32)
                colors = np.full((1, 3), 255, np.uint8)
                coverage = renderer.rasterizer.render(floor, triangles, colors, False, 2)
                assert np.array_equal(coverage[1, 1], [64, 64, 64])
                assert np.array_equal(coverage[3, 3], [255, 255, 255])
                assert np.array_equal(coverage[0, 0], [0, 0, 0])
            expected = processor.apply(image, reference_item)
            geometry_error_mean, geometry_error_fraction = 0.0, 0.0
            if args.geometry == "Taichi" and args.child != "opencv":
                geometry_reference_item = configured(params, {**config, "geometry_method": "NumPy"})
                geometry_reference = processor.apply(image, geometry_reference_item)
                geometry_error: npt.NDArray[np.float32] = np.abs(result - geometry_reference)
                geometry_error_mean = float(geometry_error.mean())
                geometry_channels: npt.NDArray[np.float32] = geometry_error.max(axis=2)
                geometry_error_fraction = (
                    float(np.count_nonzero(geometry_channels > 8)) / geometry_channels.size
                )
                assert geometry_error_mean < 1 and geometry_error_fraction < .01
                processor.retain({item["id"], reference_item["id"]})
            assert np.array_equal(image, original)
            assert result.shape == expected.shape and result.dtype == np.float32
            assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
            for _ in range(8):
                processor.apply(image, item)
            memory_before = rss_kib()
            memory_samples = []
            timings = []
            for index in range(args.frames):
                # Vary heat every frame; never time a frozen/cached result.
                current = image if index % 2 == 0 else np.roll(image, 3, axis=1)
                start = perf_counter()
                processor.apply(current, item)
                timings.append((perf_counter() - start) * 1000)
                if (index + 1) % max(1, args.frames // 4) == 0:
                    memory_samples.append({"frame": index + 1, "rss_kib": rss_kib()})
            memory_after = rss_kib()
            result = processor.apply(image, item)
            error: npt.NDArray[np.float32] = np.abs(result - expected)
            channel_error: npt.NDArray[np.float32] = error.max(axis=2)
            difference_fraction = float(np.count_nonzero(channel_error > 8)) / channel_error.size
            assert error.mean() < 5 and difference_fraction < .08
            rows.append({"detail": detail, "resolution": resolution, "shape": list(result.shape),
                         "antialiasing": args.antialiasing,
                         "geometry_method": args.geometry,
                         "geometry_mean_absolute_pixel_error": geometry_error_mean,
                         "geometry_pixels_with_error_over_8_fraction": geometry_error_fraction,
                         "reference": "original" if original_reference else "experimental OpenCV",
                         "first_frame_ms": cold_ms, "warm_median_ms": float(np.median(timings)),
                         "warm_p95_ms": float(np.percentile(timings, 95)), "frames": args.frames,
                         "effective_execution": renderer.effective_execution,
                         "mean_absolute_pixel_error": float(error.mean()),
                         "pixels_with_error_over_8_fraction": difference_fraction,
                         "rss_before_kib": memory_before, "rss_after_kib": memory_after,
                         "rss_samples": memory_samples})
            for label, output in (("result", result), ("reference", expected)):
                cv2.imwrite(str(args.output / f"{args.child}-{detail}-{label}.png"), output.astype(np.uint8))
            processor.retain(set())
        # Separate modules/processors must share initialization without sharing node-owned buffers.
        config = {**defaults, "execution": EXECUTIONS[args.child], "detail": 24, "resolution": "384",
                  "geometry_method": args.geometry}
        check_controls(processor, params, config, args.output)
        peer = CustomProcessor(apple_available=False, nvidia_available=False)
        try:
            a, b = configured(params, config), configured(params, {**config, "offset_x": 15})
            first = processor.apply(image, a)
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(processor.apply, image, a)
                fb = pool.submit(peer.apply, image, b)
                assert np.array_equal(first, fa.result(timeout=30))
                assert not np.array_equal(first, fb.result(timeout=30))
            for index in range(8):
                processor.retain(set())
                current = configured(params, {**config, "scene_rotation": (0, 90, 180, -90)[index % 4]})
                processor.apply(image, current)
                assert renderer_state(processor, current).effective_execution == config["execution"]
        finally:
            peer.close()
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python")}
        if args.child != "opencv":
            ti = importlib.import_module("taichi")

            versions["taichi"] = importlib.metadata.version("taichi")
            actual_arch = str(ti.lang.impl.current_cfg().arch)
        else:
            actual_arch = "opencv"
        return {"ok": True, "backend": args.child, "actual_arch": actual_arch,
                "package_sha256": sha256(params["package"].encode("utf-8")).hexdigest(),
                "python": sys.version, "platform": platform.platform(), "versions": versions,
                "concurrent_nodes_and_reload": True, "camera_controls_and_handedness": True,
                "cases": rows}
    finally:
        processor.close()


def main() -> int:
    """
    Bound native crashes/hangs and keep structured reports separate from native stdout/stderr.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=EXECUTIONS, default=["opencv", "metal", "cpu"])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--details", nargs="+", type=int,
                        help="Mesh details to compare at output long edge 960 (12–512)")
    parser.add_argument("--resolution", choices=("384", "640", "960", "1280", "1600", "1920",
                                               "2304", "2560", "3840", "4096"),
                        help="Override requested output long edge; four-megapixel cap still applies")
    parser.add_argument("--antialiasing", choices=("Off", "2x supersampling"), default="Off")
    parser.add_argument("--geometry", choices=("NumPy", "Taichi"), default="NumPy")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--runtime", type=Path, help="Optional isolated --target package directory")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/taichi-surface")
    parser.add_argument("--child", choices=EXECUTIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames < 1 or args.timeout < 1:
        parser.error("Frame count and timeout must be positive")
    if args.details and any(not 12 <= detail <= 512 for detail in args.details):
        parser.error("Mesh details must be between 12 and 512")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any]
    if args.child:
        try:
            report = child(args)
        except Exception as exc:  # noqa: BLE001 - Preserve arbitrary experiment failures in the report.
            # Native experiments must retain compiler/import failures in their independent report.
            traceback.print_exc()
            report = {"ok": False, "backend": args.child, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / f"{args.child}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0 if report["ok"] else 1
    if not args.runtime:
        from based_thermals.runtime_setup.taichi import ensure_runtime

        ensure_runtime(download=False)
    reports = []
    for backend in args.backends:
        environment = dict(os.environ)
        environment["TI_SKIP_VERSION_CHECK"] = "ON"
        environment["XDG_CACHE_HOME"] = str(args.output / "cache")
        if args.runtime:
            environment["PYTHONPATH"] = os.pathsep.join((str(args.runtime.resolve()), str(ROOT / "src")))
        path = args.output / f"{backend}.json"
        path.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--child", backend,
                   "--frames", str(args.frames), "--output", str(args.output)]
        if args.details:
            command.extend(["--details", *(str(detail) for detail in args.details)])
        if args.resolution:
            command.extend(["--resolution", args.resolution])
        command.extend(["--antialiasing", args.antialiasing])
        command.extend(["--geometry", args.geometry])
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                       timeout=args.timeout, check=False)
            (args.output / f"{backend}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"ok": False}
            report["exit_code"] = completed.returncode
            report["ok"] = bool(report.get("ok") and completed.returncode == 0)
        except subprocess.TimeoutExpired:
            report = {"ok": False, "error": "Native experiment timed out"}
        reports.append({"backend": backend, **report})
        print(json.dumps(reports[-1]), flush=True)
    digests = {report.get("package_sha256") for report in reports if report.get("package_sha256")}
    if len(digests) > 1:
        for report in reports:
            report["ok"] = False
            report["comparison_error"] = "Package source changed between backend processes; rerun unchanged"
    (args.output / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
