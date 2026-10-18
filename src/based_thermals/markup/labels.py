"""
Carry annotation identities and canvas transforms through deferred branch labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..custom_nodes.labels import DisplayLabel
from .model import read_annotations

IDENTITY = (1., 0., 0., 0., 1., 0., 0., 0., 1.)


@dataclass(frozen=True)
class MarkupLabel(DisplayLabel):
    """
    Keep source-local geometry separate from its accumulated normalized display transform.
    """

    node_id: str
    definition: dict[str, Any] | None = None
    matrix: tuple[float, ...] = IDENTITY

    def transformed(self, matrix: np.ndarray) -> MarkupLabel:
        """
        Move the anchor and accumulate the invertible mapping needed for on-screen editing.
        """
        anchor = matrix @ np.array((self.x, self.y, 1))
        combined = matrix @ np.asarray(self.matrix).reshape(3, 3)
        return replace(self, x=float(anchor[0] / anchor[2]), y=float(anchor[1] / anchor[2]),
                       matrix=tuple(float(v) for v in combined.ravel()))


def node_labels(item: dict[str, Any]) -> list[DisplayLabel]:
    """
    Include an invisible placement frame even when the node has no placed objects yet.
    """
    labels: list[DisplayLabel] = [MarkupLabel("", .5, .5, item["id"])]
    for definition in read_annotations(item["params"]["annotations"]):
        if definition["points"]:
            points = np.asarray(definition["points"])
            center = points.mean(axis=0)
            labels.append(MarkupLabel("", float(center[0]), float(center[1]), item["id"], definition))
    return labels


def metadata(labels: list[DisplayLabel], rotation: int) -> list[dict[str, Any]]:
    """
    Publish only small transform/identity records; current styles remain in the pipeline JSON.
    """
    from ..custom_nodes.labels import transform_labels

    return [
        {"node": label.node_id, "annotation": label.definition["id"] if label.definition else "",
         "matrix": list(label.matrix)}
        for label in transform_labels(labels, rotation) if isinstance(label, MarkupLabel)
    ]
