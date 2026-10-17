# Pipeline preset reference

[Imaging how-to](imaging-guide.md) · [Node reference](processing-pipelines.md)

## Save and load presets

Dragging a node shows a placement bar at valid drop positions in each pipeline
stack. Source and Output remain fixed, and locked stacks cannot be reordered.
The thermal-camera hardware section expands and collapses like Environment. It
starts collapsed and remembers your choice across reopenings, independently of presets
and individual node expansion. Visible-light hardware has its own disclosure when
the camera supports it, with a separate remembered choice. Collapsing either section
leaves the other section unchanged; all hardware settings remain applied.

Click the active software tab to collapse its processing section. Clicking another
tab switches to it normally; when collapsed, clicking any tab expands the section
and selects that tab. The tab row stays visible. Space or Enter on the focused tab
also toggles the section, and arrow keys switch tabs normally. Collapsing leaves
processing and individual node expansion unchanged.

The dropdown displays the loaded preset name and keeps it selected while you
edit. **Update pipeline** re-saves that preset, including pending input edits.
With no preset loaded, the button reads **Save pipeline** and asks for a name;
saving selects the new preset. **Save current pipeline as…** in the dropdown
saves under another name. Updating a bundled preset creates a user override.
Reopening Imaging settings or restarting the app preserves the loaded preset name,
including when the working pipeline has unsaved edits. Those edits remain separate
from the saved preset until you choose **Update pipeline**. With no remembered
association, matching presets are recognized automatically.
Choose **Create new** to start a blank, unsaved pipeline with only the required
source/output nodes. **Rename current pipeline…**, directly below Create new,
renames the loaded preset while preserving its saved settings and open edits.
**Delete current pipeline…** removes the loaded preset after
confirmation and keeps the open pipeline available to edit or save again. Deleted
bundled presets stay hidden across restarts without changing the shipped files.

**Import pipelines** and **Export pipelines** open checkbox lists for selecting
multiple pipelines. Export the current pipeline (including open edits), saved
presets, or both into one JSON file. Import accepts these bundles and existing
individual pipeline files, including several files at once. Name conflicts offer
Rename, Replace, or Skip; unique names are suggested to keep existing presets.
Imported pipelines are saved as presets, with an option to load the first selection.
If adding the export suffix selects an existing file, replacement requires confirmation.

Raw thermal display pixels (including raw blend/mask inputs and branches) use
the latest valid sensor frame, independently of temporal measurement averaging.
Temperature readouts, spots and graphs retain their averaged measurement plane;
calibration/invalid frames still hold the last valid image and readings.

The eight bundled presets were refreshed from the saved app collection on
2026-10-17. Each includes its saved camera controls, including explicit brightness,
contrast and gamma at 50. Bundled nodes open collapsed; the matching standalone
exports retain their saved expansion state.

**Yautja** preserves the black-hot preview, thermal-detail branch B, raw contrast
mask in C and camera-palette overlay in D, with single-pass Apple Core ML ACNet
and the saved hardware detail/noise configuration.

**Yautja GPU** bundles the current saved black-hot composition, camera-palette
overlay, contrast mask and tuned Laplacian/antialiasing/sharpening thermal branch,
with native-input Real-ESRGAN 4× using Apple Core ML `ALL` devices and camera
detail amount 50.

**Reaper Night GPU** and **Reaper Day GPU** bundle the saved masked raw-thermal
blends, gamma/contrast settings, sharpening and FFDNet denoising with Apple
Core ML `ALL` devices. Night's interpolation node is bypassed; Day has no
interpolation node. Day retains its black-hot thermal
branch, inverted mix mask and camera palette setting.

**Detail Enhanced CPU Upscale** bundles the saved Plasma preview with a Scharr
thermal-edge branch, masked subtraction, sharpening, antialiasing and
single-pass CPU ACNet enhancement with denoise level 3. Standalone exports of all
eight bundled presets are checked into `presets/`.

**Redneck Combat** is also bundled in the dropdown, preserving the original
Inferno preview with its Laplacian/contour thermal branch and masked blend.

**Redneck Combat GPU** preserves the later tuned threshold-mask blend, Scharr
edges, contours, antialiasing and sharpening in branch B, followed by native-input
Real-ESRGAN 4× in the AI enhancement node with Apple Core ML `ALL` devices.

The Pipeline presets dropdown includes **Detail Enhanced GPU Upscale**, captured
from the tuned preview/thermal-edge blend with Real-ESRGAN 4× in the current
AI enhancement node, Native input, and Apple Core ML (`ALL` compute devices).
Its complete A/B branch configuration is bundled with
the app. GPU preferences remain stored and resolve to an available GPU or CPU on other systems;
Real-ESRGAN downloads on first use if missing. Loading the preset replaces the
pipeline, not your temperature measurements or other viewer preferences.
