"""
Benchmark the portable Thermal Surface package on synthetic heat without camera access.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import node


def heat_image(width: int = 256, height: int = 192) -> np.ndarray:
    """
    Create two known peaks and a gentle ridge as a grayscale float32 thermal-like input.
    """
    y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    heat = np.clip(
        0.10 + 0.72 * np.exp(-((x + 0.25) ** 2 / 0.09 + (y - 0.15) ** 2 / 0.16))
        + 0.50 * np.exp(-((x - 0.5) ** 2 / 0.035 + (y + 0.3) ** 2 / 0.065))
        + 0.10 * np.sin(x * 8 + y * 4), 0, 1,
    )
    result = np.repeat((heat * 255).astype(np.float32)[..., None], 3, axis=2)
    return result


def main() -> int:
    """
    Report cold and warm callback timings, optionally writing input and rendered PNGs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    root = Path(__file__).resolve().parents[1]
    params = load_folder(root / "examples" / "custom_nodes" / "Thermal Surface")
    defaults = json.loads(params["config"])
    image = heat_image()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output / "input.png"), image.astype(np.uint8))
    for detail, resolution in ((24, "384"), (40, "640"), (64, "960")):
        item = node("software", "custom", **params)
        item["params"]["config"] = json.dumps({
            **defaults, "detail": detail, "resolution": resolution,
        })
        processor = CustomProcessor(apple_available=False, nvidia_available=False)
        try:
            start = perf_counter()
            result = processor.apply(image, item)
            cold = (perf_counter() - start) * 1000
            for _ in range(10):
                processor.apply(image, item)
            timings = []
            for _ in range(args.frames):
                start = perf_counter()
                result = processor.apply(image, item)
                timings.append((perf_counter() - start) * 1000)
            print(json.dumps({
                "detail": detail, "resolution": resolution, "shape": list(result.shape),
                "first_frame_ms": round(cold, 3),
                "warm_median_ms": round(float(np.median(timings)), 3),
                "warm_p95_ms": round(float(np.percentile(timings, 95)), 3),
                "frames": args.frames,
            }))
            if args.output is not None:
                cv2.imwrite(str(args.output / f"surface-{detail}.png"), result.astype(np.uint8))
        finally:
            processor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
