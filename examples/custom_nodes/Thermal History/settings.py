"""
Validate bounded profile history and orthographic view settings before state changes.
"""

from dataclasses import dataclass
from typing import Any

from based_thermals.custom_nodes.visualization import flag, number, selection


@dataclass(frozen=True)
class Settings:
    """
    Keep sampling identity separate from palette and camera changes.
    """

    profile: str
    position: float
    samples: int
    history: int
    every: int
    freeze: bool
    reset: int
    yaw: float
    tilt: float
    height: float
    scene_rotation: float
    offset_x: float
    offset_y: float
    zoom: float
    resolution: int
    wireframe: bool

    @classmethod
    def read(cls, config: dict[str, Any]) -> "Settings":
        """
        Match the literal editor defaults for standalone and embedded execution.
        """
        result = cls(
            selection(
                config,
                "profile",
                "Mean rows",
                ("Mean rows", "Mean columns", "Horizontal line", "Vertical line"),
            ),
            number(config, "position", 50, 0, 100),
            int(number(config, "samples", 96, 16, 256, integer=True)),
            int(number(config, "history", 64, 8, 256, integer=True)),
            int(number(config, "every", 1, 1, 60, integer=True)),
            flag(config, "freeze", False),
            int(number(config, "reset", 0, 0, 1000000, integer=True)),
            number(config, "yaw", -25, -180, 180),
            number(config, "tilt", 50, 15, 85),
            number(config, "height", 0.7, 0, 2),
            number(config, "scene_rotation", 0, -180, 180),
            number(config, "offset_x", 0, -100, 100),
            number(config, "offset_y", 0, -100, 100),
            number(config, "zoom", 1, 0.1, 5),
            int(
                selection(
                    config, "resolution", "1280", ("384", "640", "960", "1280", "1600", "1920")
                )
            ),
            flag(config, "wireframe", False),
        )
        return result

    def view_key(self) -> tuple[float, ...]:
        """
        Invalidate projection and floor caches only when their geometry changes.
        """
        result = (
            self.samples,
            self.history,
            self.yaw,
            self.tilt,
            self.height,
            self.scene_rotation,
            self.offset_x,
            self.offset_y,
            self.zoom,
            self.resolution,
        )
        return result
