"""
Direct pipeline annotation editing on displayed pixels, independent of thermal sampling.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ..markup.editing import MarkupSelection, handles
from ..markup.model import read_annotations, serialize
from ..markup.paint import draw_object, hit_test
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class MarkupController(SessionState):
    """
    Coordinate Markup canvas gestures through the existing separate-process UI boundary.
    """

    def markup_node(self: DesktopSession, identity: str) -> dict[str, Any] | None:
        """
        Find an enabled Markup node in any saved branch using its stable node identity.
        """
        return next((item for nodes in self.api.software_tabs(self.pipeline).values() for item in nodes
                     if item["id"] == identity and item["type"] == "markup" and not item["bypass"]), None)

    def markup_locked(self: DesktopSession, node: dict[str, Any] | None) -> bool:
        """
        Honor both explicit position locks and existing logging/calibration pipeline locks.
        """
        return node is None or node["params"]["locked"] or self.spots_state()["locked"]

    def markup_context(self: DesktopSession, identity: str) -> tuple[float, ...]:
        """
        Read an acknowledged node-to-viewer mapping from the displayed pipeline result.
        """
        entry = next((item for item in self.markup_frames if item["node"] == identity), None)
        return tuple(entry["matrix"]) if entry else ()

    def begin_markup(self: DesktopSession, command: dict[str, Any]) -> None:
        """
        Arm an existing annotation's drawing tool without altering its saved geometry.
        """
        node = self.markup_node(command.get("node", ""))
        if self.markup_locked(node):
            raise ValueError("Unlock Markup positions and stop logging/calibration before drawing")
        if self.viewer is None:
            raise ValueError("Drawing Markup requires the native viewer")
        assert node is not None
        item = next((item for item in read_annotations(node["params"]["annotations"])
                     if item["id"] == command.get("annotation")), None)
        if item is None:
            raise ValueError("Choose an existing Markup annotation")
        self.cancel_measurement_drawing()
        self.markup_selection = None
        self.markup_gesture.begin(node["id"], item, "draw", self.markup_context(node["id"]))
        self.spot_drag.cancel()
        self.picker.consume_drag_events()
        self.picker.consume_clicks()
        self.picker.consume_context_clicks()
        self.notify("Drawing Markup: " + self.markup_gesture.state()["hint"])

    def markup_objects(self: DesktopSession) -> list[tuple[str, dict[str, Any], tuple[float, ...]]]:
        """
        Resolve current definitions in stable Z order for both painting and overlap selection.
        """
        definitions = {
            node["id"]: {item["id"]: item for item in read_annotations(node["params"]["annotations"])}
            for nodes in self.api.software_tabs(self.pipeline).values() for node in nodes
            if node["type"] == "markup" and not node["bypass"]
        }
        objects = []
        for entry in self.markup_frames:
            item = definitions.get(entry["node"], {}).get(entry["annotation"])
            if item is not None:
                objects.append((entry["node"], item, tuple(entry["matrix"])))
        objects.sort(key=lambda obj: obj[1]["z_index"])
        return objects

    def reconcile_markup(self: DesktopSession) -> None:
        """
        Cancel stale gestures after deletion, bypass, locking or a canvas transform change.
        """
        gesture = self.markup_gesture
        if not gesture.mode:
            return
        node = self.markup_node(gesture.node)
        context = self.markup_context(gesture.node)
        if gesture.matrix and any(entry["node"] == gesture.node and tuple(entry["matrix"]) == gesture.matrix
                                  for entry in self.markup_frames):
            context = gesture.matrix
        if (self.markup_locked(node) or node is None
                or not any(item["id"] == gesture.identity
                           for item in read_annotations(node["params"]["annotations"]))):
            gesture.cancel()
        elif context:
            if gesture.matrix and context != gesture.matrix:
                gesture.cancel()
            else:
                gesture.matrix = context
        elif gesture.matrix:
            gesture.cancel()

    def commit_markup(self: DesktopSession) -> None:
        """
        Persist committed geometry without touching camera controls, measurements or model caches.
        """
        self.reconcile_markup()
        gesture = self.markup_gesture
        node = self.markup_node(gesture.node)
        if not gesture.mode or self.markup_locked(node):
            return
        assert node is not None
        points = gesture.geometry()
        items = deepcopy(list(read_annotations(node["params"]["annotations"])))
        for item in items:
            if item["id"] == gesture.identity:
                item["points"] = points
        node["params"]["annotations"] = serialize(items)
        if gesture.mode == "drag" and self.markup_selection is not None:
            self.markup_selection.index = gesture.handle
            if gesture.handle >= 0 and gesture.kind in ("rectangle", "textbox", "ellipse"):
                controls = handles({**gesture.item, "points": points})
                self.markup_selection.index = int(np.argmin(np.linalg.norm(
                    np.asarray(controls) - points[0], axis=1
                )))
        self.renderer.set_pipeline(self.pipeline)
        gesture.cancel()
        self.persist_settings()
        self.view_panel.update(self.view_state())

    def markup_action(self: DesktopSession, action: str) -> None:
        """
        Share native drawing shortcuts without mixing annotation and measurement state.
        """
        if action == "draw_finish":
            self.commit_markup()
        elif action == "draw_cancel":
            self.markup_gesture.cancel()
        elif action == "draw_undo":
            if self.markup_gesture.anchors:
                self.markup_gesture.anchors.pop()
            else:
                self.markup_gesture.pressed = self.markup_gesture.cursor = None

    def markup_pointer(self: DesktopSession, command: dict[str, Any]) -> bool:
        """
        Capture annotation gestures before sensor-coordinate hit testing and measurement dragging.
        """
        self.reconcile_markup()
        gesture = self.markup_gesture
        if gesture.swallow_release and not gesture.mode:
            if command["kind"] == "release":
                gesture.swallow_release = False
            elif command["kind"] == "press":
                gesture.swallow_release = False
                return False
            return True
        if not hasattr(self, "rendered"):
            return bool(gesture.mode)
        height, width = self.rendered.image.shape[:2]
        bitmap = (self.graph_layout.camera_size if self.show_graph and self.graph_layout is not None
                  else (width, height))
        x, y = command["x"], command["y"]
        position = (x / max(1, bitmap[0] - 1), y / max(1, bitmap[1] - 1))
        inside = 0 <= x < bitmap[0] and 0 <= y < bitmap[1]
        if not gesture.mode and inside and command["kind"] == "press" and command["button"] == "left":
            if self.measurement_drawing.kind is not None or self.spots.placing:
                return False
            control = self.markup_handle_hit(position, bitmap)
            selected = self.markup_selection
            candidates = list(reversed(self.markup_objects()))
            if control >= 0 and selected is not None:
                candidates = [obj for obj in candidates
                              if (obj[0], obj[1]["id"], obj[2]) ==
                              (selected.node, selected.identity, selected.matrix)]
            self.markup_selection = None
            for node_id, item, matrix in candidates:
                node = self.markup_node(node_id)
                if self.markup_locked(node):
                    continue
                assert node is not None
                pixel = round(position[0] * (width - 1)), round(position[1] * (height - 1))
                if control >= 0 or hit_test((height, width), item, matrix, pixel):
                    gesture.begin(node["id"], item, "drag", matrix)
                    gesture.handle = control
                    self.markup_selection = MarkupSelection(node_id, item["id"], control, matrix)
                    self.spot_drag.edit.cancel()
                    self.spot_drag.cancel()
                    self.picker.consume_drag_events()
                    self.picker.consume_clicks()
                    self.view_panel.update(self.view_state())
                    break
        if not gesture.mode:
            return False
        if command["button"] == "right" and command["kind"] == "release":
            if gesture.mode == "draw":
                self.commit_markup()
            return True
        if command["button"] != "left":
            return True
        point = gesture.point(position, inside=gesture.mode != "drag") if inside else None
        gesture.pointer(command["kind"], point)
        if command["kind"] == "release" and gesture.mode == "drag":
            if point is not None:
                self.commit_markup()
            else:
                gesture.cancel()
            gesture.swallow_release = False
        return True

    def draw_markup(self: DesktopSession, image: np.ndarray) -> np.ndarray:
        """
        Paint current definitions and transient drafts over the worker's unannotated display image.
        """
        self.reconcile_markup()
        gesture = self.markup_gesture
        preview = None
        objects = self.markup_objects()
        if gesture.mode:
            try:
                node = self.markup_node(gesture.node)
                definition = next((item for item in read_annotations(node["params"]["annotations"])
                                   if item["id"] == gesture.identity), gesture.item) if node else gesture.item
                preview = {**definition, "points": gesture.geometry(preview=True)}
            except ValueError:
                pass
        if preview is not None and gesture.matrix:
            replaced = False
            for index, (node_id, item, matrix) in enumerate(objects):
                if node_id == gesture.node and item["id"] == gesture.identity:
                    objects[index] = node_id, preview, matrix
                    replaced = True
            if not replaced:
                objects.append((gesture.node, preview, gesture.matrix))
            objects.sort(key=lambda obj: obj[1]["z_index"])
        if not objects:
            return image
        result = image.copy()
        for _, item, matrix in objects:
            if len(item["points"]) >= (1 if item["kind"] == "text" else 2):
                draw_object(result, item, matrix)
        return result
