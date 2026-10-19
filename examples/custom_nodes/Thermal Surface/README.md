# Thermal Surface

A portable Custom node that replaces the input image with a filled, shaded 3D
thermal height surface. Hotter input values become higher terrain, with an
orthographic camera, thermal colors, lighting, side walls and optional mesh edges.
It uses NumPy and OpenCV already included with the viewer: no GPU context, model,
download, additional dependency or background worker is needed.

The package declares `OUTPUT_KIND = "visualization"`, so when the node is directly
in pipeline A, sensor measurement graphics are hidden on this projected view.
Nodes in other tabs do not trigger hiding, even when connected to A. Measurements stay
enabled and keep supplying graph/CSV values and live Markup variables. Bypass
the node or return to a camera image feed to show the graphics again.

For a ready-made layout, select the bundled **Thermal Height Map** pipeline preset.
It embeds this package, saved view controls and processed-preview/raw insets;
no module-folder import is required. That preset uses grayscale camera Preview
intensity for elevation, with a 90° scene rotation tuned for a rotated viewer.
The Raw-source setup below is an alternative when you want heights derived from
the native temperature image; adjust Scene rotation for your viewer orientation.

## Use

1. In the software pipeline, select **Source → Raw**. Keep the image grayscale
   before this node; put any palette nodes after it or let this node supply colors.
2. Optionally add a **Range** node before it with your desired temperature limits.
   A fixed range makes heights comparable across frames. Without one, the Raw
   source's automatic temperature normalization can make terrain rise/fall as the
   scene's temperature range changes.
3. Add **Custom**, choose **Open module folder…**, and select this `Thermal Surface`
   folder. Put it before **Output**, replacing the ordinary visual map.
4. Adjust the fields in the node. Start with the defaults; for the lightest mode,
   use **Mesh detail 24** and **Output long edge 384**.

Use `Source Raw → [Range] → Custom Thermal Surface → Output` for a simple setup.
Colored input is rejected with a Raw-source hint rather than treating an arbitrary
palette's luminance as temperature. Filters before the node also affect heights.
Software palettes after the node recolor the whole rendered scene, including its
background; normally use the node's own palette field instead.

## Controls

| Field | Effect |
| --- | --- |
| Mesh detail | Samples along the longest source edge, 12–96. Default 40; more detail costs more CPU. |
| View rotation | Rotate the camera around the surface, −180° to 180°. |
| View elevation | Tilt from a low view to nearly overhead, 15°–85°. |
| Scene rotation | Roll the camera view in the image plane, −180° to 180°; positive turns the scene clockwise. Independent of orbit/elevation. |
| X offset (%) | Pan the view horizontally, −100% to 100% of output width; positive moves right. |
| Y offset (%) | Pan the view vertically, −100% to 100% of output height; positive moves down. |
| Zoom | Magnify around the fitted scene center, 0.1×–5×; default 1×. |
| Height exaggeration | Relative height, 0–2. Zero produces a flat plane. |
| Surface smoothing | Gaussian smoothing on the small sampled grid, in grid cells; zero disables it. |
| Thermal palette | All pipeline palettes and the Grayscale alias; camera-style approximations are labeled. |
| Height range | Source range retains input 0–255 scaling; Auto contrast stretches sampled 1st–99th percentiles per frame. |
| Output long edge | 384, 640 or 960 pixels. Aspect follows the input, with a minimum short edge of 32. |
| Show mesh edges | Add visible triangle borders; disabled by default for speed. |

Geometry, topology, view bounds, floor and palette are cached per node. Only the
current geometry and palette are retained; there is no growing frame history.
Changing a view/detail/output setting replaces the geometry cache. Height and
colors update every frame. Each node instance has its own caches.

Scene rotation rolls the 3D camera view rather than rotating the source heat image;
lighting, surface geometry and depth ordering stay unchanged. The rotated scene is
fitted first, then zoom and offsets are applied to both the surface and floor.
Large zoom/offset values intentionally allow the scene to extend beyond the image.
For an upright scene when the viewer itself rotates clockwise by 90°, use a scene
rotation of −90°. The new controls default to no roll/pan and 1× zoom; old JSON
configurations that omit them retain the original view when run with updated code.
Source image rows map from positive to negative world Y, preserving the input's
handedness through orbit and camera roll; at zero orbit/roll, the source's top-right
stays at the top-right of an overhead view.

## Meaning and limitations

This is **relative heat elevation**, not physical scene depth or a calibrated 3D
temperature measurement. Custom nodes receive display pixels, not the native
radiometric arrays or Celsius values. With grayscale Raw input, elevation follows
the upstream temperature-to-brightness mapping. Native measurements and recorded
temperature arrays remain unchanged.

The projected scene does not preserve sensor-pixel positions. Hide spot/dynamic
marker overlays when viewing it, and use a normal Raw/preview view to place or
interpret measurement locations. A marker drawn on the terrain is not a 3D pick.
Viewer rotation/mirroring still applies after rendering. This initial renderer uses
triangle depth sorting, not an exact per-pixel depth buffer; very steep terrain at
low elevations can have overlap artifacts. Increase elevation or reduce height.
Mesh downsampling/smoothing intentionally trades fine sensor detail for speed.

Packages embed Python/JSON source in saved/exported pipelines; no external assets
are required. Reopen the folder and resave to update an already embedded copy.
Configuration changes apply on the next processed frame without reloading code.
The renderer is original project code using the existing NumPy/OpenCV dependencies.

## Verification and benchmark

Run from the repository root with its configured Python environment:

```sh
.venv/bin/python -m pytest tests/test_thermal_surface.py
.venv/bin/python experiments/experiment_thermal_surface.py --frames 200 --output diagnostics/thermal-surface
```

The benchmark uses a synthetic 256×192 two-peak grayscale image through the real
`load_folder` / `CustomProcessor` path, including callback input copies and output
validation/conversion. It never opens a camera or changes user preferences.
Generated PNGs show the input and each quality setting. Cold first-frame cost is
reported separately from warm median/p95. Timings cover this node, not capture,
the rest of a pipeline or final Qt painting; do not treat them as whole-app FPS.

Measured on the development Mac, CPU only, 200 warm frames, wireframe off:

| Mesh detail / long edge | First frame | Warm median | Warm p95 |
| --- | ---: | ---: | ---: |
| 24 / 384 | 7.44 ms | 1.16 ms | 1.55 ms |
| 40 / 640 (default) | 8.27 ms | 2.85 ms | 3.39 ms |
| 64 / 960 | 11.95 ms | 6.97 ms | 8.55 ms |

These are synthetic local results, not live-camera or Linux performance claims.
Tests cover portable save/export, known projection geometry, changing heights,
cache isolation/replacement, flat/tiny/portrait inputs, invalid configuration,
colored-input rejection, and real offscreen editor interactions through refresh,
rebuild, export and locks.

## Full palette catalog

The palette dropdown now includes every software pipeline palette, plus the
legacy Grayscale alias. Camera-style entries marked **(approx.)** use the same
software approximations as the palette node, not exact hardware LUTs. Existing
Inferno/Turbo/Grayscale values and defaults remain compatible. Updated packages
use `based_thermals.custom_nodes.visualization.palette_lut`; no GPU is initialized
by palette selection. Reopen this folder and save the pipeline to replace an
older embedded package; opening a folder loads its defaults.
