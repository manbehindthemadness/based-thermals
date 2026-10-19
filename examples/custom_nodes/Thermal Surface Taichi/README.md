# Thermal Surface Taichi experiment

This separate **Custom** package compares the original Thermal Surface renderer
with a compute-only Taichi rasterizer. It creates no windows, opens no camera,
and adds no shared viewer backend, project dependency or preset. Taichi CPU is
selected by default, with OpenCV fallback when Taichi is unavailable. The existing
Thermal Surface and bundled Thermal Height Map preset are unchanged.

Select Source **Raw**, then load this folder with **Custom → Open module folder…**.
The experimental renderer dropdown selects OpenCV, Taichi CPU, Metal, Vulkan or
CUDA. All other controls have the original meaning. Only Apple M4 Metal and ARM
CPU execution have been measured; CUDA/Vulkan are unverified experiment choices.
This is relative display-brightness elevation, not physical depth or calibrated
temperature data. `OUTPUT_KIND="visualization"` preserves normal A-only hiding
of measurement graphics, without disabling sampling or Markup readouts.

The current defaults match the selected viewer setup: mesh 256, output long edge
1280, view rotation 80°, elevation 45°, scene rotation 90°, X offset −14%, Y offset
0%, zoom 1.0, height 0.7, smoothing 3.0, Turbo palette, Source range, mesh edges off
and 2x supersampling on. Reopen this folder and save to update an embedded node.

## Optional isolated setup

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
To try the folder in the viewer later, its process also needs the optional package
directory on `PYTHONPATH`; otherwise choosing Taichi produces OpenCV fallback.
Alternatively, install it into the viewer's existing environment:

```sh
uv pip install --python .venv/bin/python taichi==1.7.4
```

Restart the viewer after installation. This optional local installation is not
part of project dependencies; an exact `uv sync` may remove it, requiring the
command again. The node needs a **Source → Raw** input before any color palette
or visualization node. The missing-runtime fallback cannot correct colorized
input and deliberately retains that validation error.

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

Both renderers use the same NumPy geometry, smoothing, normalization, per-face
lighting, palette and stable far-to-near triangle ordering. The Taichi path
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

The evidence supports further dense-mesh experiments, including the Taichi CPU
path. It does not justify making Taichi a required dependency or replacing the
existing default renderer.
