"""
Transient normalized drawing and dragging gestures with no measurement dependencies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .editing import handles, move_handle


@dataclass
class MarkupGesture:
    """
    Keep canvas previews outside persistent pipeline settings until explicit completion.
    """

    node: str = ""
    identity: str = ""
    kind: str = ""
    mode: str = ""
    matrix: tuple[float, ...] = ()
    item: dict[str, Any] = field(default_factory=dict)
    anchors: list[list[float]] = field(default_factory=list)
    pressed: list[float] | None = None
    cursor: list[float] | None = None
    swallow_release: bool = False
    released: bool = False
    handle: int = -1

    def begin(self, node: str, item: dict[str, Any], mode: str,
              matrix: tuple[float, ...] = ()) -> None:
        """
        Arm a drawing tool or a drag using a snapshot of the annotation's original geometry.
        """
        self.cancel()
        self.swallow_release = False
        self.node, self.identity, self.kind, self.mode = node, item["id"], item["kind"], mode
        self.item = deepcopy(item)
        self.matrix = matrix

    def cancel(self) -> None:
        """
        Release temporary geometry and swallow the remainder of a cancelled mouse gesture.
        """
        self.swallow_release = self.swallow_release or (self.pressed is not None and not self.released)
        self.node = self.identity = self.kind = self.mode = ""
        self.item = {}
        self.anchors.clear()
        self.pressed = self.cursor = None
        self.matrix = ()
        self.released = False
        self.handle = -1

    def point(self, position: tuple[float, float], *, inside: bool = True) -> list[float] | None:
        """
        Invert the branch/viewer transform to edit coordinates in the node's own canvas.
        """
        if not self.matrix:
            return None
        try:
            inverse = np.linalg.inv(np.asarray(self.matrix).reshape(3, 3))
        except np.linalg.LinAlgError:
            return None
        moved = inverse @ np.array((*position, 1))
        if abs(moved[2]) < 1e-9:
            return None
        point = moved[:2] / moved[2]
        if not np.isfinite(point).all() or inside and (np.any(point < 0) or np.any(point > 1)):
            return None
        return [float(value) for value in point]

    def pointer(self, kind: str, point: list[float] | None) -> None:
        """
        Update a drag, box/line gesture, clicked label or bounded polygon draft.
        """
        if point is not None and (not self.released or self.kind == "polygon"):
            self.cursor = point
        if kind == "press":
            self.released = False
            self.cursor = point
            if self.mode == "draw" or self.pressed is None:
                self.pressed = point
        elif kind == "release" and self.mode == "draw":
            self.released = True
            if point is None:
                self.pressed = None
                return
            if self.kind == "polygon":
                if self.pressed is not None:
                    if len(self.anchors) >= 64:
                        raise ValueError("Markup polygons support up to 64 corners")
                    self.anchors.append(point)
                self.pressed = None
            elif self.kind == "text":
                self.pressed = point

    def geometry(self, *, preview: bool = False) -> list[list[float]]:
        """
        Return a complete normalized object or a preview without changing its saved definition.
        """
        if self.mode == "drag":
            if self.pressed is None or self.cursor is None:
                return deepcopy(self.item["points"])
            source = np.asarray(self.item["points"])
            delta = np.asarray(self.cursor) - self.pressed
            if self.handle >= 0:
                origin = np.asarray(handles(self.item)[self.handle])
                return move_handle(self.item, self.handle, (origin + delta).tolist())
            delta = np.clip(delta, -source.min(axis=0), 1 - source.max(axis=0))
            return (source + delta).tolist()
        if self.kind == "polygon":
            points = deepcopy(self.anchors)
            if preview and self.cursor is not None:
                points.append(self.cursor)
            if len(points) < (2 if preview else 3):
                raise ValueError("Click at least three corners, then Finish")
            return points
        if self.pressed is None or self.cursor is None:
            raise ValueError("Click for a label, or drag on the image to draw")
        if not preview and self.mode == "draw" and not self.released:
            raise ValueError("Release the mouse before finishing")
        if self.kind == "text":
            return [self.pressed.copy()]
        if not preview and np.linalg.norm(np.asarray(self.cursor) - self.pressed) < 1e-4:
            raise ValueError("Drag between two distinct points")
        return [self.pressed.copy(), self.cursor.copy()]

    def state(self) -> dict[str, Any]:
        """
        Publish contextual finish/undo/cancel controls to the native viewer.
        """
        finish = False
        try:
            self.geometry()
            finish = True
        except ValueError:
            pass
        return {"active": self.mode == "draw", "role": "Markup", "kind": self.kind,
                "hint": "Click corners; Enter to finish" if self.kind == "polygon"
                else "Click for text or drag a shape; Enter to finish",
                "finish": finish, "undo": bool(self.anchors or self.pressed)}
