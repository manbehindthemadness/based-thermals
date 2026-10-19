"""
Render camera-free previews and benchmark the three portable thermal visualizations.
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

FOLDERS = Path(__file__).resolve().parents[1] / "examples" / "custom_nodes"


def frame(index: int) -> np.ndarray:
    """
    Generate moving Gaussian hot spots and a cooler wave on a 256×192 grayscale plane.
    """
    y, x = np.mgrid[-1:1:192j, -1:1:256j].astype(np.float32)
    shift = np.sin(index / 18) * 0.25
    plane = 25 + 165 * np.exp(-((x + 0.3 + shift) ** 2 / 0.12 + (y + 0.15) ** 2 / 0.3))
    plane += 100 * np.exp(-((x - 0.5) ** 2 / 0.08 + (y - 0.3 - shift) ** 2 / 0.13))
    plane += 20 * (np.sin(x * 8 + y * 3 + index / 12) + 1)
    result = np.repeat(np.clip(plane, 0, 255)[:, :, None], 3, axis=2)
    return result


def run(output: Path, frames: int) -> dict[str, dict[str, float]]:
    """
    Save a contact sheet, individual images, history video and warm/cold CPU timings.
    """
    output.mkdir(parents=True, exist_ok=True)
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    results: dict[str, dict[str, float]] = {}
    previews = []
    names = ("Thermal Contours", "Thermal Shaded Relief", "Thermal History")
    try:
        for name in names:
            item = node("software", "custom", **load_folder(FOLDERS / name))
            timings = []
            movie = None
            last_result: np.ndarray | None = None
            try:
                for index in range(max(frames, 80)):
                    source = frame(index)
                    start = perf_counter()
                    current = processor.apply(source, item)
                    timings.append((perf_counter() - start) * 1000)
                    last_result = current
                    if name == "Thermal History":
                        preview = preview_image(current)
                        if movie is None:
                            movie = cv2.VideoWriter(
                                str(output / "thermal-history.mp4"),
                                cv2.VideoWriter.fourcc(*"mp4v"),
                                25,
                                (640, 480),
                            )
                            if not movie.isOpened():
                                raise RuntimeError("Could not create the history preview video")
                        movie.write(preview)
                assert last_result is not None
                result = last_result
                filename = name.lower().replace(" ", "-")
                if not cv2.imwrite(
                    str(output / f"{filename}.png"), result.round().astype(np.uint8)
                ):
                    raise RuntimeError(f"Could not save {filename}")
                panel = preview_image(result)
                heading = np.full((40, 640, 3), 18, np.uint8)
                cv2.putText(
                    heading,
                    name,
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                previews.append(np.vstack((heading, panel)))
                warm = np.array(timings[64:])
                results[name] = {
                    "cold_ms": timings[0],
                    "warm_median_ms": float(np.median(warm)),
                    "warm_p95_ms": float(np.percentile(warm, 95)),
                }
            finally:
                if movie is not None:
                    movie.release()
        cv2.imwrite(str(output / "comparison.png"), np.hstack(previews))
        (output / "timings.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    finally:
        processor.close()
    return results


def preview_image(image: np.ndarray) -> np.ndarray:
    """
    Materialize OpenCV's resized output as ordinary NumPy pixels for PNG/video previews.
    """
    resized = np.asarray(cv2.resize(image, (640, 480)), dtype=np.float32)
    result = np.rint(resized).astype(np.uint8)
    return result


def main() -> None:
    """
    Select an output directory and at least eighty frames, without opening camera hardware.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/thermal-visualizations"))
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if args.frames < 80:
        parser.error("--frames must be at least 80")
    print(json.dumps(run(args.output, args.frames), indent=2))


if __name__ == "__main__":
    main()
