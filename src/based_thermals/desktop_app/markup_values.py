"""
Read-only annotation snapshots from published temperatures, cached measurements and session metrics.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..markup.variables import finite, pipeline_value, references
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class MarkupValuesController(SessionState):
    """
    Build small primitive snapshots without querying camera hardware or changing radiometry.
    """

    def update_markup_values(self: DesktopSession) -> None:
        """
        Refresh only active template dependencies, sharing the existing frame-scoped readings.
        """
        now = self.api.time.monotonic()
        self.markup_runtime.observe(now, self.frame_captured_at if self.fresh_frame else None)
        values = self.markup_runtime.values(now)
        local = datetime.now(UTC).astimezone()
        values.update({"Clock.Time": local.strftime("%H:%M:%S"), "Clock.Date": local.strftime("%Y-%m-%d"),
                       "Temperature.Unit": self.renderer.temperature_unit,
                       "Timing.Pipeline": self.pipeline_elapsed_ms,
                       "Camera.NominalFPS": self.camera_profile.frame_rate,
                       "Measurement.Valid": bool(self.rendered.measurements_valid
                                                 and self.rendered.radiometry_available),
                       "Measurement.Status": self.rendered.measurement_status,
                       "Recording.Elapsed": self.recorder.elapsed_seconds(now=now),
                       "Recording.Frames": self.recorder.frames_written})
        valid = self.rendered.measurements_valid and self.rendered.radiometry_available
        spatial = valid and self.rendered.measurement_mapping_valid
        values["Measurement.Valid"] = bool(spatial)
        values["Temperature.Valid"] = bool(valid)
        fahrenheit = self.renderer.temperature_unit == "F"

        def temperature(value: float) -> float | None:
            """
            Convert display readouts only; all stored and sampled temperatures remain Celsius.
            """
            converted = value * 1.8 + 32 if fahrenheit else value
            return converted if valid and math.isfinite(converted) else None

        for name in ("minimum", "maximum", "average", "center"):
            values[f"Temperature.{name.title()}"] = temperature(getattr(self.rendered.stats, name))
        required = references(self.pipeline)
        if any(key.startswith("Measurement.") and key not in values for key in required) and spatial:
            readings = self.spots.display_cache.readings(self.spots, self.rendered.temperatures_celsius)
            for number, reading in readings.items():
                for field in ("minimum", "maximum", "average"):
                    values[f"Measurement.{number}.{field.title()}"] = temperature(getattr(reading, field))
                values[f"Measurement.{number}.Count"] = reading.count
        for reference in required:
            if reference.startswith("Pipeline/"):
                values[reference] = finite(pipeline_value(self.pipeline, reference))
        self.markup_runtime.metrics = any(key.startswith("Display.") or key in ("Timing.Draw", "Timing.Total")
                                          for key in required)
        self.markup_live = {key: finite(value) for key, value in values.items()}

    def markup_metrics(self: DesktopSession, command: dict[str, Any]) -> None:
        """
        Accept bounded, throttled paint statistics from the existing native viewer JSON channel.
        """
        for source, key, limit in (("fps", "Display.FPS", 10000), ("frames", "Display.Frames", 1e12),
                                   ("draw_ms", "Timing.Draw", 1e7), ("total_ms", "Timing.Total", 1e9)):
            value = command.get(source)
            if value is None:
                self.markup_runtime.viewer[key] = None
            elif (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and 0 <= value <= limit and math.isfinite(value)):
                self.markup_runtime.viewer[key] = value
        self.markup_runtime.viewer_at = self.api.time.monotonic()
