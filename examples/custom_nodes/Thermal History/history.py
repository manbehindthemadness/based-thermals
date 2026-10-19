"""
Retain a bounded ring of brightness profiles, independently for each Custom node.
"""

from typing import Any

import cv2
import numpy as np

from based_thermals.custom_nodes.visualization import display_settings, grayscale, normalized

from .mesh import HistoryMesh
from .settings import Settings


class HistoryRenderer:
    """
    Store at most 256×256 float32 values, never full camera frames or unbounded lists.
    """

    def __init__(self) -> None:
        """
        Start empty with no devices, timers or independent threads.
        """
        self.buffer = np.empty((0, 0), np.float32)
        self.count = 0
        self.head = 0
        self.tick = 0
        self.sampling_key: tuple[Any, ...] | None = None
        self.mesh: HistoryMesh | None = None
        self.view_key: tuple[float, ...] | None = None

    def render(self, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """
        Record processed-frame profiles, then render oldest-to-newest with a shared heat range.
        """
        colors, mode, smoothing = display_settings(config)
        settings = Settings.read(config)
        heat = grayscale(image, smoothing)
        key = (
            heat.shape,
            settings.profile,
            settings.position,
            settings.samples,
            settings.history,
            settings.reset,
            smoothing,
        )
        if self.sampling_key != key:
            self.buffer = np.empty((settings.history, settings.samples), np.float32)
            self.count = self.head = self.tick = 0
            self.sampling_key = key
        view = settings.view_key()
        mesh = self.mesh
        if mesh is None or self.view_key != view:
            mesh = HistoryMesh(settings)
            self.mesh = mesh
            self.view_key = view
        if not settings.freeze:
            if self.tick % settings.every == 0:
                profile = sample_profile(heat, settings)
                self.buffer[self.head] = profile
                self.head = (self.head + 1) % settings.history
                self.count = min(self.count + 1, settings.history)
            self.tick += 1
        indices = (np.arange(self.count) + self.head - self.count) % settings.history
        history = self.buffer[indices]
        if self.count:
            history = normalized(history, mode)
        result = mesh.render(history, colors, settings.wireframe)
        return result


def sample_profile(heat: np.ndarray, settings: Settings) -> np.ndarray:
    """
    Reduce a row, column or spatial average into a fixed-width relative-brightness profile.
    """
    if settings.profile == "Mean rows":
        profile = heat.mean(axis=0)
    elif settings.profile == "Mean columns":
        profile = heat.mean(axis=1)
    elif settings.profile == "Horizontal line":
        index = round((heat.shape[0] - 1) * settings.position / 100)
        profile = heat[index]
    else:
        index = round((heat.shape[1] - 1) * settings.position / 100)
        profile = heat[:, index]
    result = cv2.resize(
        profile.reshape(1, -1), (settings.samples, 1), interpolation=cv2.INTER_AREA
    )[0]
    return result
