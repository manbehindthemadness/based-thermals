"""
Selection handles and keyboard editing for annotation-only pipeline objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ..markup.editing import MarkupSelection, handles
from ..markup.paint import project
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class MarkupEditController(SessionState):
    """
    Keep display editing independent of native measurement selection and sampling.
    """

    def select_markup(self: DesktopSession, command: dict[str, Any]) -> None:
        """
        Select an object from the expanded element list without arming a drawing tool.
        """
        node = self.markup_node(command.get("node", ""))
        self.markup_gesture.cancel()
        self.markup_selection = None
        if node is None or self.markup_locked(node):
            return
        from ..markup.model import read_annotations

        if any(item["id"] == command.get("annotation")
               for item in read_annotations(node["params"]["annotations"])):
            self.markup_selection = MarkupSelection(node["id"], command["annotation"])
            self.spot_drag.edit.cancel()

    def selected_markup(self: DesktopSession) -> tuple[dict[str, Any], tuple[float, ...]] | None:
        """
        Reconcile selection against deletion, locks, branch visibility and transformed instances.
        """
        selected = self.markup_selection
        if selected is None:
            return None
        if self.markup_locked(self.markup_node(selected.node)):
            self.markup_selection = None
            return None
        for node_id, item, matrix in self.markup_objects():
            if ((node_id, item["id"]) == (selected.node, selected.identity)
                    and (not selected.matrix or selected.matrix == matrix)):
                selected.matrix = matrix
                if selected.index >= len(handles(item)):
                    selected.index = -1
                return item, matrix
        self.markup_selection = None
        return None

    def markup_handle_hit(self: DesktopSession, position: tuple[float, float],
                          bitmap: tuple[int, int]) -> int:
        """
        Hit only the selected object's visible controls using a fixed screen-pixel tolerance.
        """
        current = self.selected_markup()
        if current is None:
            return -1
        item, matrix = current
        points = handles(item)
        if not points:
            return -1
        mapped = project(points, matrix)
        distance = np.linalg.norm((mapped - position) * np.maximum(1, np.asarray(bitmap) - 1), axis=1)
        index = int(np.argmin(distance))
        return index if distance[index] <= 8 else -1

    def markup_nudge(self: DesktopSession, direction: object) -> bool:
        """
        Nudge the selected object/control one display pixel along the viewer's oriented axes.
        """
        if self.markup_selection is None:
            return False
        current = self.selected_markup()
        delta = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.get(
            direction if isinstance(direction, str) else ""
        )
        if current is None or delta is None or self.markup_gesture.mode or not hasattr(self, "rendered"):
            return True
        selected = self.markup_selection
        assert selected is not None
        item, matrix = current
        origin = handles(item)[selected.index] if selected.index >= 0 else item["points"][0]
        screen = project([origin], matrix)[0]
        height, width = self.rendered.image.shape[:2]
        moved = screen + np.asarray(delta) / (max(1, width - 1), max(1, height - 1))
        gesture = self.markup_gesture
        gesture.begin(selected.node, item, "drag", matrix)
        gesture.handle = selected.index
        gesture.pressed = list(origin)
        gesture.cursor = gesture.point((float(moved[0]), float(moved[1])), inside=False)
        if gesture.cursor is not None:
            self.commit_markup()
        else:
            gesture.cancel()
        gesture.swallow_release = False
        return True

    def draw_markup_handles(self: DesktopSession, image: np.ndarray) -> np.ndarray:
        """
        Draw measurement-style yellow selection handles only in the interactive viewer overlay.
        """
        current = self.selected_markup()
        if current is None or self.markup_gesture.mode == "draw":
            return image
        item, matrix = current
        if self.markup_gesture.mode == "drag":
            item = {**item, "points": self.markup_gesture.geometry(preview=True)}
        controls = handles(item)
        if not controls:
            return image
        mapped = project(controls, matrix) * (image.shape[1] - 1, image.shape[0] - 1)
        result = image.copy()
        selected = self.markup_selection
        assert selected is not None
        for index, point in enumerate(mapped):
            if np.isfinite(point).all() and np.all(np.abs(point) < 65536):
                pixel = tuple(round(value) for value in point)
                radius = 4 if index == selected.index else 3
                cv2.circle(result, pixel, radius, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(result, pixel, radius, (0, 220, 255), 2 if radius == 4 else 1, cv2.LINE_AA)
        return result
