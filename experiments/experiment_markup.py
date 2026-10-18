"""
Measure actual display annotation compositing on synthetic frames without camera access.
"""

import argparse
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np

from based_thermals.desktop_app.session import DesktopSession
from based_thermals.markup.interaction import MarkupGesture
from based_thermals.markup.labels import IDENTITY
from based_thermals.markup.model import KINDS, annotation, serialize
from based_thermals.pipeline import default_pipeline, node, software_tabs


def main() -> None:
    """
    Report cold/warm display costs and optionally save a preview of all supported tools.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error("--frames must be from 1 to 10000")
    yy, xx = np.indices((576, 768))
    source = np.asarray(cv2.applyColorMap(((xx + yy) % 256).astype(np.uint8), cv2.COLORMAP_INFERNO))
    for count in (1, 10, 128, 0):
        items = []
        for index in range(count):
            kind = KINDS[index % len(KINDS)]
            item = annotation(kind)
            x, y = .02 + (index % 8) * .12, .03 + (index // 8) * .055
            item["points"] = ([[x, y]] if kind == "text" else
                              [[x, y], [x + .1, y], [x + .05, y + .04]] if kind == "polygon"
                              else [[x, y], [x + .1, y + .04]])
            item["text"] = "Stage" if kind in ("text", "textbox") else ""
            items.append(item)
        if count == 0:
            table = annotation("table")
            table.update(points=[[.05, .05], [.8, .8]],
                         text="Metric\tValue\nPeak\t{{ [Temperature.Maximum] :.1f }}\n"
                              "Frames\t{{ [Capture.Frames] :d }}\nTime\t{{ [Session.Elapsed] :.1f }}")
            items.append(table)
        markup = node("software", "markup", annotations=serialize(items))
        document = default_pipeline()
        document["software"].insert(1, markup)
        controller = DesktopSession.__new__(DesktopSession)
        controller.api = SimpleNamespace(software_tabs=software_tabs)
        controller.pipeline = document
        controller.markup_frames = [
            {"node": markup["id"], "annotation": item["id"], "matrix": list(IDENTITY)} for item in items
        ]
        controller.markup_gesture = MarkupGesture()
        controller.markup_live = {}
        timings = []
        result = source
        for frame_index in range(args.frames + 1):
            controller.markup_live = {"Temperature.Maximum": 25 + frame_index / 10,
                                      "Capture.Frames": frame_index, "Session.Elapsed": frame_index / 25}
            started = perf_counter()
            result = controller.draw_markup(source)
            timings.append((perf_counter() - started) * 1000)
        median, p95 = float(np.median(timings[1:])), float(np.percentile(timings[1:], 95))
        label = f"{count:3d} objects" if count else "Live table"
        print(f"{label}: cold {timings[0]:.3f} ms; median {median:.3f} ms; p95 {p95:.3f} ms")
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output / f"markup-{count}.png"), result):
                raise OSError("Unable to write Markup preview")


if __name__ == "__main__":
    main()
