# Thermal Surface Taichi experiment

## Deferred work: GPU presentation on a separate branch

Recorded 2026-10-19. Finish the current visualizations first. Investigate GPU
presentation later on a dedicated branch, tentatively
`feature/gpu-visualization-renderer`. Implementation is deferred until the
current visualization work is finished.

The hypothesis is that avoiding finished-image readback and CPU image copies
could reduce latency. Readback has not been isolated as the dominant remaining
cost. On Apple Silicon, unified memory still requires synchronization and may
involve conversions/copies. Merely replacing the image canvas with OpenGL would
retain the current Custom node's NumPy output and subsequent CPU processing.

Proposed design:

- Define a common scene description: scalar heat grid, camera/view settings,
  palette, lighting and display annotations.
- Add an optional GPU presentation path using Qt's rendering abstraction:
  Metal on macOS; tested Vulkan/OpenGL capabilities on Linux, including NVIDIA.
  Graphics acceleration must not depend on CUDA being installed.
- Keep the existing NumPy/OpenCV rasterizer and Qt image canvas as an explicit,
  supported pure-CPU path. Qt Quick's software backend does not support all
  shader features, so it cannot substitute for this implementation.
- Initially send the small heat grid and scene settings to the viewer process
  and let that process own GPU resources. Avoid cross-process sharing of Taichi
  buffers/textures in the first prototype.
- Keep requested backend preferences, provide capability-based fallback and an
  explicit CPU override. Verify installed PySide6 bindings and optional package
  requirements before choosing a specific Qt API.

Preserve the separate Qt/OpenCV process boundary, latest-frame behavior, native
radiometry, measurement sampling and Markup readouts. Rotation/mirroring, pointer
mapping, annotation editing/locks/Z order and A-only visualization hiding must
remain consistent. CPU image output must remain available for downstream image
nodes, branch combinations, capture and recording; define where materialization
is required before changing the Custom node's ndarray-only contract. A terminal
height-map view is the initial prototype scope.

First measure GPU computation, synchronization/readback, CPU conversions/copies,
frame transport and presentation separately. Then compare end-to-end latency,
cold startup, frame pacing, memory and visual fidelity on Apple Silicon,
Linux/NVIDIA and forced CPU operation. Keep real GPU/camera checks outside normal
pytest collection. Revisit implementation after the visualization work is finished.

References: [Qt graphics architecture](https://doc.qt.io/qt-6/topics-graphics.html),
[Qt software limitations](https://doc.qt.io/qt-6/qtquick-visualcanvas-adaptations-software.html),
[Metal shared storage](https://developer.apple.com/documentation/metal/mtlstoragemode/shared),
[Apple OpenGL migration guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-macos-apps-to-apple-silicon).

## Current experiment

This separate **Custom** package compares the original Thermal Surface renderer
with a compute-only Taichi rasterizer. It creates no windows, opens no camera,
and adds no shared viewer backend or preset. Taichi CPU is
selected by default, with OpenCV fallback when Taichi is unavailable. The existing
Thermal Surface and bundled Thermal Height Map preset retain their rendering defaults.

Select Source **Raw**, then load this folder with **Custom → Open module folder…**.
The experimental renderer dropdown selects OpenCV, Taichi CPU, Metal, Vulkan or
CUDA. All other controls have the original meaning. Apple M4 Metal/ARM CPU and
NVIDIA GB10 Linux ARM64 CPU/Vulkan execution have been measured; CUDA is unverified.
This is relative display-brightness elevation, not physical depth or calibrated
temperature data. `OUTPUT_KIND="visualization"` preserves normal A-only hiding
of measurement graphics, without disabling sampling or Markup readouts.

The current defaults match the selected viewer setup: mesh 256, output long edge
1280, view rotation 80°, elevation 45°, scene rotation 90°, X offset −14%, Y offset
0%, zoom 1.0, height 0.7, smoothing 3.0, Turbo palette, Source range, mesh edges off
and 2x supersampling on. Reopen this folder and save to update an embedded node.

## Compiled geometry experiment

**Geometry + lighting → Taichi** moves vertex projection, face normals, lighting,
palette lookup and face gathering into compiled kernels on the selected Taichi
CPU/Metal backend. The default **NumPy** keeps the previous path for comparison.
The OpenCV renderer always uses NumPy geometry. Heat resizing, smoothing and
normalization remain in NumPy/OpenCV; stable painter-depth sorting remains in
NumPy to preserve overlap behavior. This is separate from choosing the renderer.

The compiled path retains one current mesh's topology, vertices and intermediate
arrays per node. Each frame uploads scalar heat and reads back only face depths
for sorting; triangle/color arrays stay in the Taichi runtime through rasterization.
The small palette, camera and sorted index arrays also cross the kernel boundary.
It adds memory and initial compilation cost, and is not bit-identical to NumPy's
floating-point arithmetic. No shared application backend or camera code changes.
Missing runtimes and catchable compiler errors retain the existing OpenCV fallback.
Switching Geometry + lighting clears a previous failure so that selection can retry.

The kernel source guard now covers both raster and mesh kernels. Restart the
viewer when loading this update after a Taichi runtime has already initialized,
then reopen/resave the folder snapshot to obtain the new control. Existing
snapshots without the setting use NumPy geometry.

Run the camera-free comparison twice, changing only `--geometry` and `--output`:

```sh
.venv/bin/python experiments/experiment_taichi_surface.py \
  --backends cpu metal --frames 150 --details 40 128 256 512 \
  --resolution 1280 --antialiasing '2x supersampling' --geometry NumPy \
  --output diagnostics/taichi-geometry-numpy-final
.venv/bin/python experiments/experiment_taichi_surface.py \
  --backends cpu metal --frames 150 --details 40 128 256 512 \
  --resolution 1280 --antialiasing '2x supersampling' --geometry Taichi \
  --output diagnostics/taichi-geometry-compiled
```

Measured 2026-10-19 on the M4 setup documented below, using the screenshot view,
1280×960 output, 2x supersampling, 8 warmups and 150 changing frames per case.
Times include the full Custom callback, copies, validation, sorting and completed
readback. NumPy and compiled runs used identical package SHA256
`0ceedc5e66a067d948ca27bbca10a9b798fc784c54bce5b5fe4e35ec9daf4ede`.

| Mesh | CPU NumPy → compiled median | Metal NumPy → compiled median |
| --- | ---: | ---: |
| 40 | 13.38 → 13.19 ms | 7.73 → 8.54 ms |
| 128 | 19.46 → 20.29 ms | 11.54 → 11.59 ms |
| 256 | 41.38 → 23.71 ms | 28.39 → 16.46 ms |
| 512 | 109.51 → 48.82 ms | 93.84 → 32.97 ms |

Dense meshes benefit; small meshes do not show a consistent gain. At 256, p95
CPU callback time changed 62.41→37.45 ms and Metal 46.38→31.75 ms. First compiled
frame was 601 ms CPU/481 ms Metal versus 524/243 ms for NumPy geometry; cold times
include import, JIT and allocation. Sequential runs are affected by desktop load,
and some IDE/focused checks overlapped portions of the runs. These are indicative
callback comparisons, not live camera-to-screen latency or a controlled hardware
benchmark. Full application tests ran afterward.

At 256, compiled versus NumPy geometry with the *same* rasterizer differed by
0.00000271/255 mean channel value on CPU and 0.00000678 on Metal, with no pixels
differing by more than 8. Independent projected-coordinate/depth checks passed,
along with orientation, pan/zoom, portrait, wireframe, concurrent node and reload
checks. The compiled PNG was inspected. Floating-point differences may be larger
for other views. RSS checkpoints fluctuate with host/native allocator pressure;
these short runs do not establish leak freedom. JSON includes p95, memory and
separate OpenCV-raster comparison errors. No live camera was used.

### NumPy regression investigation, 2026-10-19

`experiments/experiment_taichi_regression.py` snapshots the committed pre-geometry
package without changing the checkout, then runs old/new/new/old in fresh native
processes. A separate process measures NumPy→Taichi→NumPy twice. Each phase has
20 warmups and 150 changing frames; separate instrumentation reports heat,
lighting and raster medians outside the main timings. It validates the effective
backend and exact output equality when returning to NumPy. `--config` accepts a
plain settings JSON object and applies it identically to both versions.

With Source range at the screenshot defaults, old/current CPU median-of-run-medians
was 35.11/34.98 ms. Metal's repeated old and current ranges overlapped; an initial
small difference reversed on repetition. No persistent geometry-addition slowdown
or fallback was reproduced. Mode switching sometimes added 1–2 ms, without
progressive growth. Host load varied and the existing viewer remained running;
these synthetic callbacks isolate the node rather than measuring that viewer.

Saved settings additionally used Auto contrast and height 0.2. Auto contrast's
percentile arithmetic returned float64 heat, widening NumPy's downstream mesh
temporaries. That predated the new geometry option. The normalized heat now
returns to float32 before mesh construction, while percentile normalization
retains its existing arithmetic. Source-range rendering is unchanged; an
independent original-renderer Auto-contrast check allows under 0.05 mean channel
error and verifies normalized heat agreement within 2e-7.

Paired CPU Auto-contrast runs measured 37.79–39.92 ms before and 32.13–32.73 ms
after, approximately 17% lower callback time. Lighting stages changed from
23.07–23.51 ms to 17.82–18.02 ms. This fixes an existing memory/precision cost,
not evidence of a new NumPy regression. No native temperature arrays change.
The source-range, repeated Metal, original Auto-contrast and fixed Auto-contrast
reports are respectively in `diagnostics/taichi-regression-initial`,
`diagnostics/taichi-regression-metal-repeat`,
`diagnostics/taichi-regression-auto-contrast` and
`diagnostics/taichi-regression-auto-contrast-fixed`. The fixed package SHA256 is
`bc76340e58e61e3ad34eefb95032c69d526a50557bf669f7cb8c6fbd4b7c3b79`.

```sh
.venv/bin/python experiments/experiment_taichi_regression.py \
  --reference 65ce5da --frames 150 --output diagnostics/taichi-regression
```

A direct float16 probe at mesh details 256 and 512 underflowed representative
face-normal lengths to zero, as well as the `1e-8` normalization floor; float32
returned nonzero lengths. Float16 is therefore not used for geometry arithmetic.
Mixed float16 storage/float32 computation would require separate performance and
image checks; no float16 speedup has been measured.

## Installation

Normal project setup installs the viewer dependencies:

```sh
uv sync --dev
```

Viewer startup separately installs optional Taichi 1.7.4 into an application cache
on CPython 3.12–3.13 Apple Silicon macOS and x86-64 Linux, matching the
[published wheels](https://pypi.org/project/taichi/1.7.4/#files). Taichi is absent
from `pyproject.toml` and `uv.lock`; normal exact `uv sync` cannot remove the cached
runtime. Python 3.14+ and unsupported/offline environments retain OpenCV fallback.
GPU initialization can also fall back to OpenCV. See
[runtime setup](../../../docs/taichi-runtime-setup.md) for retry and offline controls.

Linux ARM has no official 1.7.4 PyPI wheel and never automatically compiles Taichi;
use the [Taichi build skill](../../../.agents/skills/build-taichi/SKILL.md) for an
opt-in ARM64 CPU + Vulkan source installation:

```sh
.venv/bin/python .agents/skills/build-taichi/scripts/build_taichi.py
```

Run that command from the project root after installing the native prerequisites
listed in the skill. The helper builds a wheel, verifies a hardware Vulkan kernel
in staging, and registers it in the separate cache with a verified CPU backend.
Choose **Taichi CPU** or **Taichi Vulkan** after restarting the viewer. CPU support
is required for compatibility; unavailable Vulkan never prevents CPU registration.
OpenCV remains an independent CPU fallback. The build omits CUDA.
Ordinary `uv sync` preserves this runtime. Wheel, source provenance and native logs
stay in `diagnostics/taichi-cpu-vulkan-build/`; the active runtime lives in the user cache.

On 2026-10-19, the patched LLVM source build passed real CPU/Vulkan kernels and the
camera-free surface experiment on Ubuntu 24.04 ARM64,
CPython 3.12.3 and NVIDIA GB10 (driver 580.173.02). See the
[measured scope](../../../.agents/skills/build-taichi/references/build-notes.md#measured-scope-2026-10-19)
for provenance, the ARM atomics fix and the remaining CUDA build limitation.

### Isolated runtime experiments

From the repository root, using the configured Python 3.12 SDK and uv:

```sh
UV_CACHE_DIR=diagnostics/taichi-uv-cache uv pip install \
  --python .venv/bin/python --target diagnostics/taichi-runtime --no-deps \
  taichi==1.7.4 colorama==0.4.6 dill==0.4.1 rich==15.0.0 \
  markdown-it-py==4.2.0 mdurl==0.1.2 pygments==2.21.0
.venv/bin/python experiments/experiment_taichi_surface.py \
  --runtime diagnostics/taichi-runtime --frames 1000 \
  --output diagnostics/taichi-surface-verified
```

This installs optional packages only under ignored diagnostics and reuses the
project's NumPy/OpenCV. It does not modify `.venv`, `pyproject.toml` or `uv.lock`.
The runner supplies that directory to disposable children through `PYTHONPATH`.
This isolated setup is only for comparing runtime builds; the viewer normally
uses the dependency installed by `uv sync`. The node needs a **Source → Raw**
input before any color palette or visualization node. The missing-runtime fallback
cannot correct colorized input and deliberately retains that validation error.

Native GPU compilation must be allowed by the execution environment. The initial
restricted Metal probe failed shader compilation; it passed with normal native
GPU access and a writable diagnostic cache. No driver or OS changes were needed.

The implementation is original project code; geometry, settings and OpenCV
fallback are adapted from the sibling Thermal Surface with its arithmetic and
triangle ordering retained, keeping this package portable when embedded. Taichi
is an external Apache-2.0 dependency, not bundled
source/binaries. Published build and reference APIs:
[Taichi 1.7.4](https://pypi.org/project/taichi/1.7.4/),
[array interoperability](https://docs.taichi-lang.org/docs/ndarray).
Reopen this folder and resave a pipeline to replace an older embedded snapshot.

## Dense meshes

Mesh detail accepts 12–512 samples along the input's long edge; the default
is 256. At 4:3, detail 256 produces a 256×192 grid and detail 512 produces
512×384. Values above the input dimensions interpolate brightness rather than
adding sensor information. Output resolution remains a separate control.
Output long-edge choices are 384, 640, 960, 1280, 1600, 1920, 2304, 2560,
3840 and 4096. The default is 1280. Requests exceeding the pipeline's
4,000,000-pixel limit scale down both dimensions together: a 4:3 request of 4096
renders at 2309×1732, a square at 2000×2000, and sufficiently wide inputs can
reach the full 4096 long edge. Integer rounding can slightly change aspect ratio.
Higher output sizes increase rasterization, transfer and display costs without
changing mesh detail or sensor resolution.
The **Antialiasing** control offers **Off** and **2x supersampling** (default).
Supersampling keeps subpixel triangle coordinates and averages four coverage
samples into each output pixel, smoothing silhouettes and wireframe edges.
It preserves output dimensions and radiometry; coarse mesh facets still require
more mesh detail or surface smoothing. Taichi retains a four-times-larger owner
buffer but resolves directly into the final color image. OpenCV fallback renders
a temporary 2×-per-axis image and area-downsamples it. Both cost extra memory and
raster work; the final output still respects the four-megapixel cap.
Existing embedded nodes must reopen this folder and resave to acquire the new
controls and runtime validation.
Decimal controls display and accept one decimal place; mesh detail remains an
integer. Numeric keypad drafts survive live state updates and apply on Enter,
focus loss or save/export. The height exaggeration step is 0.1.
This antialiasing update changes the native kernels, so restart the viewer after
reopening the folder if Taichi has already initialized. Preserve current values
before reopening: folder import replaces them with package defaults.

The camera-free dense run used 8 warmups and 50 changing frames per case at
960×720 output on the M4 setup below. Median callback times were:

| Detail | OpenCV | Taichi CPU | Metal |
| --- | ---: | ---: | ---: |
| 128 | 18.79 ms | 14.81 ms | 8.90 ms |
| 256 | 66.32 ms | 26.12 ms | 25.45 ms |
| 512 | 294.52 ms | 102.85 ms | 107.16 ms |

These short measurements overlapped application tests and are indicative under
load, not controlled speed comparisons. At 512, CPU/Metal mean channel error
against this package's OpenCV renderer was 0.0182/255; 0.00854% of pixels differed
by more than 8. The original renderer remains capped at 96, so higher-detail
comparisons use the experimental OpenCV path. Handedness, view controls,
concurrent nodes and reload checks passed. No live camera was used.
Reports, memory samples and PNGs: `diagnostics/taichi-surface-dense`;
package SHA256 `ea0d6b45992e16611c508d3769ab1107ebb868be7f1027d3f0d18edb1d5a2105`.
Reproduce with:

```sh
.venv/bin/python experiments/experiment_taichi_surface.py \
  --runtime diagnostics/taichi-runtime --frames 50 --details 128 256 512 \
  --output diagnostics/taichi-surface-dense
```

Add `--resolution 4096` to exercise the larger output request with the same
four-megapixel cap. High-resolution comparisons use this package's OpenCV path
because the original example only accepts output edges through 960.

### Antialiasing verification

The camera-free `--antialiasing '2x supersampling'` experiment passed on OpenCV,
Taichi CPU and Metal at detail 128/output 960. Thirty-frame warm medians were
22.07, 12.12 and 10.74 ms respectively; these short samples are indicative.
Reports and PNGs are under `diagnostics/taichi-surface-aa` (package SHA256
`d6ee3fbda7ce0ceecc8ea00364abe7b54fcd4512628e615e85ce2d2014ba57dd`,
before an import-order-only cleanup). An independent fractional triangle check
verified quarter coverage as 64/255, full interior as 255 and background as 0
on both native backends. CPU/Metal mean channel error against supersampled
OpenCV was 0.0887/255; 0.1055% of pixels differed by more than 8. Raster coverage
still differs slightly between backends. The resulting surface PNG was inspected.
Both native backends also passed a maximum-output request (4096, capped to
2309×1732 for 4:3) with antialiasing enabled; see
`diagnostics/taichi-surface-aa-large`. Disabled-antialiasing native checks against
the original renderer passed separately in `diagnostics/taichi-surface-aa-off`.
These final smoke reports use package SHA256
`f19fae61d6b8bb59869cdb5764a380fd9a6a6c049887c69f88d83d5f7f5b1637`.

## What is being compared

With Geometry + lighting set to NumPy, both renderers use the same NumPy geometry,
smoothing, normalization, per-face lighting, palette and stable far-to-near triangle ordering. The Taichi path
parallelizes triangle coverage and resolves overlap with integer atomic painter
ranks, then returns a BGR NumPy image. It does **not** introduce a depth buffer or
change the existing low-elevation overlap limitations. Triangle-edge coverage
and wireframe line widths differ slightly from OpenCV's rasterization; degenerate
projected triangles are skipped. Use OpenCV when exact coverage matching matters.

Each node retains only its current raster owner buffer and original geometry
cache. One small kernel library and a lock are shared through the imported
Taichi module across this package's independent node namespaces. There are no
node-owned background threads, fields or GUI objects. Native execution is
serialized; per-node buffers remain private. Removing/reloading nodes releases
their arrays; the kernel library/runtime stay alive until process exit.

The node never calls `ti.reset()`. A different already-initialized Taichi backend
or external runtime reset causes OpenCV fallback; restart to change Taichi's
backend. A changed `kernels.py` digest also requires a restart, with OpenCV
fallback until then, so reloading source cannot silently reuse stale kernels.
Other configuration and CPU helper edits use the normal package reload behavior.
It disables Taichi's version-check network request by default and uses
no offline kernel cache. Initial JIT compilation still incurs a noticeable cost.
Catchable import/compiler/device errors log a warning once per selected mode
and use the original surface thereafter. Select OpenCV and reselect a Taichi mode
to retry. Native aborts/hangs cannot be contained by Python callback fallback;
the benchmark runs each backend in a child with a timeout and separate JSON/log
files. The experimental node has not been validated during live USB capture.

## Measured results, 2026-10-18

Apple M4 (10-core GPU), macOS 26.6.2 ARM64, Python 3.12.5, Taichi 1.7.4
(`b4b956fd`, LLVM 15.0.7), NumPy 2.5.3, OpenCV 5.0.0.93. Native backend evidence
was `Arch.metal` and `Arch.arm64`, with explicit fallback disabled. Inputs were
synthetic 256×192 grayscale heat, shifted on alternate frames, wireframe off.
Each case used 8 warmups and 1,000 timed frames through `CustomProcessor`, including
input copying, output validation/conversion, GPU synchronization and host readback.
These are node callback times, not camera-to-screen latency or whole-app FPS.

| Detail / long edge | OpenCV median / p95 | Taichi CPU median / p95 | Metal median / p95 |
| --- | ---: | ---: | ---: |
| 24 / 384 | 1.11 / 1.32 ms | 0.95 / 1.09 ms | 1.31 / 1.75 ms |
| 40 / 640 | 2.61 / 2.98 ms | 1.85 / 2.16 ms | 2.38 / 3.08 ms |
| 64 / 960 | 5.95 / 6.57 ms | 3.68 / 4.21 ms | 4.26 / 5.18 ms |
| 96 / 960 | 11.18 / 12.02 ms | 5.21 / 5.88 ms | 5.97 / 7.21 ms |

First Taichi frame, including import/init/JIT: 319 ms CPU and 264 ms Metal.
OpenCV's first case took 19.6 ms. All three reports have package SHA256
`dc3bfa05797f6588d914fcced5a5d16f3456a8d3b9dc95d8cf3a62f0ff03f7d7`.
The runner rejects comparisons with mixed package digests. Earlier 100/1000-frame
samples showed different gains at default detail; timing varies with host load.
A second Metal experiment
retaining device-side floor/output buffers measured 7.82 ms median at 96/960 and
showed no benefit; that extra buffering is not included in this package.

At 96/960 the Taichi/OpenCV mean absolute channel difference was 0.118 on a
0–255 scale; 0.277% of pixels differed by more than 8 in any channel. At 24/384
the corresponding figures were 0.267 and 2.06%. CPU and Metal reported the same
comparison errors. PNGs were visually inspected. These differences are display
raster edges, not changes to native sensor readings.

The native runs also passed independent hot-corner handedness checks at four
camera rolls, portrait sizing, wireframe/pan/zoom changes, two concurrent node
instances and eight removal/reload cycles. Process RSS grows while native
allocators warm: at 96/960 Metal ranged approximately 278,192–278,512 KiB
from frames 250–1000; Taichi CPU ranged 296,464–296,656 KiB. Earlier/larger
allocation steps occur at quality changes, so this short run is not proof of
leak freedom or prompt return of native memory to the OS. JSON reports include
quarter-run RSS checkpoints outside timed sections.

Final reports/PNGs/logs are under `diagnostics/taichi-surface-verified`; earlier
1000-frame runs are under `diagnostics/taichi-surface-long`. Earlier
100-frame reports are under `diagnostics/taichi-surface-measured`, and the rejected
buffering trial is under `diagnostics/taichi-surface-buffered`. Diagnostics are
ignored outputs. Normal tests fake the optional boundary and perform no installs,
device initialization or network requests:

```sh
.venv/bin/python -m pytest tests/test_taichi_surface_package.py tests/test_thermal_surface.py
```

Taichi is now a project dependency on environments with published compatible
wheels. These measurements support the contained Custom-node renderer; the
existing viewer renderer is retained.

## Full palette catalog

All software pipeline palettes are now available in both geometry paths and
OpenCV/Taichi rasterizers, including White/Black hot and the legacy Grayscale
alias. Camera-style entries marked **(approx.)** use the pipeline's software
approximations, not exact hardware LUTs. They share the same 256×3 BGR ramp through
`based_thermals.custom_nodes.visualization.palette_lut`; palette selection does
not initialize a device. Existing defaults and saved palette values are unchanged.
Reopen this folder and save the pipeline to update an older embedded package;
opening a folder loads its defaults.

Camera-free verification on 2026-10-19 rendered all 22 choices on native Taichi CPU
with both NumPy and Taichi geometry, without fallback. OpenCV tests cover all five
visualization packages; this palette update was not separately run on Metal,
Vulkan, CUDA or Linux hardware.
