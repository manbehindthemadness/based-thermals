# Thermal Contours

Add a **Custom** software node and open this folder. Place **Source → Raw** and
optionally a **Range** node before it. No extra dependencies or GPU are required.

The node displays relative-heat contour boundaries and optional filled color
bands. Defaults: 12 intervals, Turbo, smoothing 1, one-pixel dynamic lines at
70% opacity. Choose a fixed RGB line color or `dynamic` to use the measurement
contrast compositor: inverted strokes with a black/white halo that remains visible
on middle gray. Disable bands for a continuous palette; disable lines for bands
alone. Interval count is 2–64 and line width is 1–5 input-image pixels.

The app's `palette` control provides all software palettes and saved Palette builder
gradients. The shared visualization settings helper resolves the app's cached LUT;
the package contains no palette catalog or color table. Camera-style palettes use
software approximations, not exact camera LUTs. Legacy labels such as Grayscale
remain accepted; new defaults use stable palette identifiers.

Contours divide display brightness into equal intervals. They are **relative
heat contours**, not calibrated Celsius/Fahrenheit isotherms. Source range uses
0–255; Auto contrast uses the current image's 1st/99th percentiles. Geometry and
native measurements remain unchanged, and measurement graphics stay visible.

Settings and source are embedded when saving a pipeline. Reopen the module folder
and save again to update an older embedded version; opening a folder loads its
defaults. There are no external assets or retained frame history.

For synthetic previews and CPU timings, run the repository's
`experiments/experiment_thermal_visualizations.py`. Tests cover the real portable
runtime, palette parity, band boundaries, ownership, JSON export and editor locks.
