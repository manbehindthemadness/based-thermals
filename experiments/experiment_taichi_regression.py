"""
Compare committed and working NumPy geometry in fresh processes and after native-mode switching.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from experiment_taichi_surface import (
    EXECUTIONS,
    FOLDER,
    ROOT,
    configured,
    heat_image,
    renderer_state,
    rss_kib,
)

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor


def measure(processor: CustomProcessor, item: dict[str, Any], image: np.ndarray,
            frames: int) -> dict[str, Any]:
    """
    Include callback copies, validation and completed native readback after twenty changing warmups.
    """
    shifted = np.roll(image, 3, axis=1)
    for index in range(20):
        processor.apply(image if index % 2 == 0 else shifted, item)
    timings = []
    memory_before = rss_kib()
    for index in range(frames):
        current = image if index % 2 == 0 else shifted
        start = perf_counter()
        processor.apply(current, item)
        timings.append((perf_counter() - start) * 1000)
    renderer = renderer_state(processor, item)
    result = {"median_ms": float(np.median(timings)),
              "p95_ms": float(np.percentile(timings, 95)), "frames": frames,
              "effective_execution": renderer.effective_execution,
              "fallback_reason": renderer.fallback_reason,
              "rss_before_kib": memory_before, "rss_after_kib": rss_kib()}
    return result


def profile(processor: CustomProcessor, item: dict[str, Any], image: np.ndarray) -> dict[str, float]:
    """
    Measure CPU heat/lighting and synchronous raster separately, outside uninstrumented frame timings.
    """
    module = sys.modules[processor.packages[item["id"]].name]
    namespace = module.__package__
    assert namespace is not None
    surface = sys.modules[namespace + ".surface"]
    samples: dict[str, list[float]] = {key: [] for key in ("heat", "lighting", "raster")}
    originals = surface.heat_values, surface.lit_triangles, surface.Rasterizer.render

    def wrapped(function: Any, label: str) -> Any:
        """
        Retain each stage's result while timing only its existing work.
        """
        def timed(*values: Any, **options: Any) -> Any:
            """
            Record one stage duration without changing buffers or arguments.
            """
            start = perf_counter()
            result = function(*values, **options)
            samples[label].append((perf_counter() - start) * 1000)
            return result
        return timed

    surface.heat_values = wrapped(originals[0], "heat")
    surface.lit_triangles = wrapped(originals[1], "lighting")
    surface.Rasterizer.render = wrapped(originals[2], "raster")
    try:
        for index in range(20):
            processor.apply(image if index % 2 == 0 else np.roll(image, 3, axis=1), item)
    finally:
        surface.heat_values, surface.lit_triangles, surface.Rasterizer.render = originals
    return {key + "_median_ms": float(np.median(values))
            for key, values in samples.items() if values}


def child(args: argparse.Namespace) -> dict[str, Any]:
    """
    Run exactly one source/backend in this native process; never open camera hardware or preferences.
    """
    params = load_folder(args.folder)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config.update(execution=EXECUTIONS[args.backend], geometry_method="NumPy")
    item = configured(params, config)
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    image = heat_image()
    phases = []
    try:
        modes = (("NumPy", "Taichi", "NumPy", "Taichi", "NumPy")
                 if args.child == "switch" else ("NumPy",))
        first = None
        for method in modes:
            item["params"]["config"] = json.dumps({**config, "geometry_method": method})
            start = perf_counter()
            result = processor.apply(image, item)
            first_ms = (perf_counter() - start) * 1000
            if method == "NumPy":
                if first is None:
                    first = result.copy()
                else:
                    np.testing.assert_array_equal(result, first)
            measured = measure(processor, item, image, args.frames)
            assert measured["effective_execution"] == config["execution"], measured
            measured.update(geometry=method, first_frame_ms=first_ms)
            if method == "NumPy":
                measured["stages"] = profile(processor, item, image)
            phases.append(measured)
        return {"ok": True, "source": args.child, "backend": args.backend,
                "package_sha256": sha256(params["package"].encode()).hexdigest(),
                "config": config, "load_average": os.getloadavg(), "phases": phases}
    finally:
        processor.close()


def snapshot(revision: str, destination: Path) -> None:
    """
    Materialize only the reference package's Python/JSON blobs without changing the working checkout.
    """
    relative = FOLDER.relative_to(ROOT)
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", revision, "--", str(relative)],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    for name in listed.stdout.splitlines():
        path = Path(name).relative_to(relative)
        if path.suffix not in (".py", ".json"):
            continue
        source = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.stdout, encoding="utf-8")


def main() -> int:
    """
    Bound native subprocesses and alternate old/new source order to expose load and initialization effects.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="65ce5da")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "metal"), default=["cpu", "metal"])
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/taichi-regression")
    parser.add_argument("--child", choices=("old", "new", "switch"), help=argparse.SUPPRESS)
    parser.add_argument("--folder", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help="JSON settings applied identically to both sources")
    parser.add_argument("--backend", choices=("cpu", "metal"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames < 1 or args.reference.startswith("-"):
        parser.error("Frame count must be positive and reference must name a commit")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any]
    if not args.child:
        from based_thermals.runtime_setup.taichi import ensure_runtime

        ensure_runtime(download=False)
    if args.child:
        try:
            report = child(args)
        except Exception as exc:  # noqa: BLE001 - Retain native experiment errors and traceback.
            traceback.print_exc()
            report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0 if report["ok"] else 1
    revision = subprocess.run(["git", "rev-parse", "--verify", f"{args.reference}^{{commit}}"],
                              cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    config = (json.loads(args.config.read_text(encoding="utf-8")) if args.config
              else json.loads(load_folder(FOLDER)["config"]))
    if not isinstance(config, dict):
        parser.error("Configuration must be a JSON object")
    path = args.output / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    reports = []
    with tempfile.TemporaryDirectory(prefix="reference-", dir=args.output) as temporary:
        reference = Path(temporary)
        snapshot(revision, reference)
        for backend in args.backends:
            for index, label in enumerate(("old", "new", "new", "old", "switch")):
                output = args.output / f"{backend}-{index}-{label}"
                output.mkdir(parents=True, exist_ok=True)
                environment = {**os.environ, "TI_SKIP_VERSION_CHECK": "ON",
                               "XDG_CACHE_HOME": str(args.output / "cache")}
                command = [sys.executable, str(Path(__file__).resolve()), "--child", label,
                           "--folder", str(reference if label == "old" else FOLDER),
                           "--config", str(path), "--backend", backend,
                           "--frames", str(args.frames), "--output", str(output)]
                try:
                    result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                            check=False, timeout=120)
                    (output / "native.log").write_text(result.stdout + result.stderr, encoding="utf-8")
                    result_path = output / "result.json"
                    report = json.loads(result_path.read_text()) if result_path.exists() else {"ok": False}
                    report.update(source=label, backend=backend, exit_code=result.returncode)
                    report["ok"] = bool(report["ok"] and result.returncode == 0)
                except subprocess.TimeoutExpired:
                    report = {"ok": False, "source": label, "backend": backend, "error": "Timed out"}
                reports.append(report)
                print(json.dumps(report), flush=True)
    summary = {"reference": revision, "reports": reports}
    (args.output / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all(row["ok"] for row in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
