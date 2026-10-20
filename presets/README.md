# Pipeline exports

These nine files capture saved app presets, refreshed on 2026-10-17 with
**Thermal Height Map** added on 2026-10-18. Import them from
**Imaging → Pipeline → Import**. The same processing
settings are bundled in `src/based_thermals/predefined_pipeline_presets.json`, so
installed users see them in the preset dropdown without local preference files.

The original eight presets were refreshed from the saved app collection on 2026-10-17.
They include the saved camera processing, boost, palette, detail and noise settings,
plus explicit brightness, contrast and gamma controls at 50. Their saved software
branches, enhancement passes, model parameters and bypass states are retained.
**Detail Enhanced GPU Upscale** retains detail at 91, boost mode 3, camera palette
10, three-pass GPU RealESR enhancement and its tuned raw Scharr edge branch.

- `Detail Enhanced CPU Upscale.pipeline.json`
- `Detail Enhanced GPU Upscale.pipeline.json`
- `Reaper Day GPU.pipeline.json`
- `Reaper Night GPU.pipeline.json`
- `Redneck Combat.pipeline.json`
- `Redneck Combat GPU.pipeline.json`
- `Yautja.pipeline.json`
- `Yautja GPU.pipeline.json`
- `Thermal Height Map.pipeline.json`

**Thermal Height Map** includes the portable Thermal Surface Custom package,
so selecting the preset needs no module-folder import, downloads or extra dependencies.
It retains the saved mesh detail 60, Turbo palette, 960-pixel output, view position,
camera controls (including contrast 25), and processed-preview/raw inset branches.
Refreshed from the saved preset on 2026-10-18, it also includes the locked Markup
node with inset labels, two arrows, and a surface caption using dynamic contrast.
The main surface uses grayscale camera Preview intensity as relative elevation;
it is a visualization, not physical depth or a calibrated temperature axis.
Its scene rotation of 90° is tuned for a rotated viewer; adjust the Custom node's
Scene rotation for your display orientation. Sensor measurement overlays are
automatically hidden when the surface node is directly in pipeline A; sampling,
graphs and Markup variables remain active. See the [Thermal Surface example](../examples/custom_nodes/Thermal%20Surface/README.md).

On 2026-10-19 its embedded source and palette dropdown were refreshed to include
all software pipeline palettes, with camera-style approximations labeled explicitly.
Its saved configuration, layout and other nodes were preserved.

On 2026-10-20 its two Markup labels were given explicit source-local centers using
the preset's original 270° portrait layout. Their positions no longer depend on
whether the viewer starts at 0°, 90°, 180° or 270°. Each label stays beside its
arrow tail in either portrait orientation. Arrow geometry, text, styles and camera
settings are unchanged. To update an already saved copy, restart the app and select **Thermal Height Map** again, or import the refreshed export. Reloading
the Custom module or performing another factory reset is unnecessary.

Exports retain saved node identities, expansion and bypass states, hardware
controls, every software branch, and model/backend settings. Bundled presets use
collapsed nodes throughout; processing settings and bypass states match the exports.
Optional models prepare on first use; saved GPU preferences retain the normal CPU
fallback when the requested backend is unavailable.

Fresh Duo installations start with **Detail Enhanced CPU Upscale**. Existing
installations restore their saved working pipeline, including unsaved preset edits.
Other camera profiles retain their own startup and capability rules. Updating
these exports does not modify local preferences or include unrelated viewer settings.
