# Thermal Shaded Relief

Add a **Custom** software node and open this folder after **Source → Raw** and an
optional **Range** node. This top-down hillshade uses NumPy/OpenCV on CPU; no
additional dependencies, models or assets are needed.

Defaults: Turbo, smoothing 1, light azimuth 315°, elevation 45°, exaggeration 6,
ambient light 0.3 and relief mix 1. Change azimuth to illuminate slopes from the
opposite direction; elevation controls grazing versus overhead light. Ambient
light sets the minimum illumination. Mix 0 returns the unshaded palette image.

The app's `palette` control provides all software palettes and saved Palette builder
gradients. The shared visualization settings helper resolves the app's cached LUT;
the package contains no palette catalog or color table. Legacy labels such as
Grayscale remain accepted; new defaults use stable identifiers. Camera-style
palettes use software approximations, not exact hardware LUTs. Source range maps
display values 0–255;
Auto contrast uses the current image's 1st/99th percentiles.

This is a lighting visualization of relative brightness, not physical surface
geometry or a temperature conversion. It preserves image coordinates and leaves
measurement sampling and graphics active. The implementation uses float32 Sobel
gradients with image-size scaling and retains no images between frames.

Saving a pipeline embeds the configuration and Python source. Reopen this folder
and save again to replace an older embedded package; opening a folder loads its
defaults. The camera-free preview/benchmark experiment is
`experiments/experiment_thermal_visualizations.py`. Tests include an independent
flat-plane lighting result, opposing slopes, palette parity and the real editor.
