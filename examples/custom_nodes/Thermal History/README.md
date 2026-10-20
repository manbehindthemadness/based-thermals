# Thermal History

Add a **Custom** software node and open this folder after **Source → Raw** and an
optional **Range** node. This CPU-only waterfall surface needs NumPy/OpenCV,
already included in the application. It does not require Taichi or a GPU.

Each processed frame contributes a horizontal or vertical brightness profile.
**Mean rows** averages all rows into a horizontal profile; **Mean columns**
averages all columns into a vertical profile. **Horizontal line** and **Vertical
line** sample the selected line position from 0–100% of the corresponding axis.
The surface's left-to-right axis follows increasing profile position. Its depth
runs from older profiles at the back to newest profiles at the front, before
applying view rotation.

Defaults: 96 profile samples, 64 history samples, record every frame, Turbo,
smoothing 1, 1280×960 output, view rotation −25°, elevation 50°, height 0.7.
View rotation/elevation, scene rotation, pan, zoom and mesh edges are editable.
Output long edge ranges from 384–1920 at fixed 4:3 aspect, below four megapixels.
Profile and history detail each range up to 256; higher detail increases raster
cost. Edges use OpenCV antialiasing.

**Record every N processed frames** controls sampling cadence. This is a history
of processing callbacks, not a clock or a guarantee of distinct camera captures.
At 25 processed frames/s, 64 samples recorded every frame cover about 2.56 seconds;
slower processing or a different cadence changes that span. No timestamps or
temperature arrays are passed to Custom nodes.

**Freeze history** stops recording while allowing palette and camera changes.
Change **Clear history** to clear the ring once. Changing input shape, profile
source/position, sample count, history length or smoothing also clears it.
Changing palette, normalization, view, mesh edges or cadence preserves history.
An empty frozen history shows only the floor, without invented zero-valued samples.
State is private to each node and is discarded on node unload/reload or app exit.
Only settings/source are saved, not the recorded history.

The ring stores at most 256×256 float32 values (256 KiB), rather than full frames.
Source range uses 0–255. Auto contrast computes one 1st/99th-percentile range over
the recorded window, so all profiles use the same scale. Changing an upstream
Range changes display brightness; clear history after such a change if you need
a consistent comparison. Height and color represent relative brightness, never
physical elevation or calibrated temperatures.

The app's `palette` control provides all software palettes and saved Palette builder
gradients. The shared visualization settings helper resolves the app's cached LUT;
the package contains no palette catalog or color table. Legacy labels such as
Grayscale remain accepted; new defaults use stable identifiers. Camera-style
palettes use software approximations, not exact camera LUTs.
Measurement graphics automatically hide only when this visualization
is directly in pipeline A; measurement sampling, graphs and Markup variables
remain active. Other tabs follow the viewer's existing visibility rules.

Save the pipeline to embed the Python package and settings. Reopen this folder
and save again to update an older package; opening a folder loads its defaults.
Tests cover bounded rings, independent node state, cadence, freeze/reset, profile
sampling, palette parity, export and editor locks. The camera-free experiment
`experiments/experiment_thermal_visualizations.py` writes a contact sheet, an MP4
history preview and CPU timings. This prototype does not change the viewer's
renderer or implement the deferred GPU presentation architecture.
