"""
Guarded object clipboard, context menus and navigation across the Qt JSON boundary.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..markup.clipboard import encode, paste
from ..markup.editing import MarkupSelection
from ..markup.model import read_annotations, serialize
from ..markup.paint import hit_test
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class MarkupCommands(SessionState):
    """
    Apply display-only object edits without resetting hardware, models or measurement history.
    """

    def markup_target(self: DesktopSession, command: dict[str, Any]) -> dict[str, Any] | None:
        """
        Resolve an explicit menu target, then selection, then the last visible unlocked node.
        """
        identity = command.get("node")
        if identity is not None:
            return self.markup_node(identity) if isinstance(identity, str) else None
        if self.markup_selection is not None:
            return self.markup_node(self.markup_selection.node)
        for entry in reversed(self.markup_frames):
            node = self.markup_node(entry["node"])
            if not self.markup_locked(node):
                return node
        return None

    def markup_command(self: DesktopSession, command: dict[str, Any]) -> None:
        """
        Validate current locks and identities before copying, pasting, deleting or editing.
        """
        operation = command.get("operation")
        if operation not in ("copy", "paste", "delete", "edit") or self.markup_gesture.mode == "draw":
            return
        node = self.markup_target(command)
        if node is None or self.markup_locked(node):
            return
        items = deepcopy(list(read_annotations(node["params"]["annotations"])))
        identity = command.get("annotation")
        if (identity is None and self.markup_selection is not None
                and self.markup_selection.node == node["id"]):
            identity = self.markup_selection.identity
        item = next((item for item in items if item["id"] == identity), None)
        if operation != "paste" and item is None:
            return
        if operation == "copy":
            assert item is not None
            self.markup_response = {"serial": uuid4().hex, "clipboard": encode(item)}
            return
        if operation == "edit":
            assert item is not None
            self.markup_gesture.cancel()
            self.markup_selection = MarkupSelection(node["id"], item["id"])
            self.spot_drag.edit.cancel()
            self.spot_drag.cancel()
            node["expanded"] = True
            self.markup_focus = {"serial": uuid4().hex, "node": node["id"], "annotation": item["id"]}
            self.persist_settings()
            self.view_panel.open(self.view_state())
            return
        if operation == "paste":
            added = paste(items, command.get("text"))
            self.markup_selection = MarkupSelection(node["id"], added["id"])
            self.spot_drag.edit.cancel()
            self.spot_drag.cancel()
            # The new object shares this node's already acknowledged projection until worker refresh.
            frames = [entry for entry in self.markup_frames if entry["node"] == node["id"]]
            for entry in frames:
                mapped = {**entry, "annotation": added["id"]}
                if mapped not in self.markup_frames:
                    self.markup_frames.append(mapped)
        else:
            items = [item for item in items if item["id"] != identity]
            self.markup_selection = None
        self.markup_gesture.cancel()
        node["params"]["annotations"] = serialize(items)
        self.renderer.set_pipeline(self.pipeline)
        self.persist_settings()
        self.view_panel.update(self.view_state())

    def markup_context_pointer(self: DesktopSession, command: dict[str, Any]) -> bool:
        """
        Hit test the displayed Z order for right-click menus or double-click editor navigation.
        """
        if self.markup_gesture.mode or self.measurement_drawing.kind is not None or self.spots.placing:
            return False
        if not hasattr(self, "rendered"):
            return False
        height, width = self.rendered.image.shape[:2]
        bitmap = (self.graph_layout.camera_size if self.show_graph and self.graph_layout is not None
                  else (width, height))
        x, y = command["x"], command["y"]
        if not (0 <= x < bitmap[0] and 0 <= y < bitmap[1]):
            return False
        pixel = round(x / max(1, bitmap[0] - 1) * (width - 1)), round(y / max(1, bitmap[1] - 1) * (height - 1))
        hit = None
        for node_id, item, matrix in reversed(self.markup_objects()):
            if not self.markup_locked(self.markup_node(node_id)) and hit_test((height, width), item, matrix, pixel):
                self.markup_selection = MarkupSelection(node_id, item["id"], matrix=matrix)
                hit = {"node": node_id, "annotation": item["id"]}
                break
        if command["kind"] == "double":
            if hit is None:
                return False
            self.markup_command({**hit, "operation": "edit"})
            self.markup_gesture.swallow_release = True
        else:
            node = self.markup_target(hit or {})
            if node is None or self.markup_locked(node):
                return False
            self.spot_drag.edit.cancel()
            self.spot_drag.cancel()
            self.markup_response = {"serial": uuid4().hex, "menu": {
                "node": node["id"], "annotation": hit["annotation"] if hit else None,
            }}
            self.view_panel.update(self.view_state())
        self.picker.consume_context_clicks()
        self.picker.consume_clicks()
        self.picker.consume_drag_events()
        return True
