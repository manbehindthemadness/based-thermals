# External resource checks

Run the model checks below explicitly from the repository root using the configured project
environment. They are outside pytest's normal `tests/` collection. The startup
capture below explicitly opens the camera; the model and synthetic checks do not.

## Live startup-frame capture

Close other camera viewers, then launch the normal desktop through this passive recorder:

```sh
.venv/bin/python experiments/experiment_startup_frames.py
# Optional desktop arguments follow --:
.venv/bin/python experiments/experiment_startup_frames.py -- --rotate 90
```

On macOS, enter the normal USB helper password in the launching terminal if
requested. Let the startup image settle and leave the app open for at least
12 seconds, then quit normally. The app keeps its usual hardware startup and
settings restoration; the recorder adds no camera commands or second USB owner.
Files are written only after normal cleanup, under a new timestamped directory in
`diagnostics/startup-frames` (`--output` changes the parent directory).

The first 32 assembled frames are retained densely, then every fourth frame,
bounded to 12 seconds, 96 samples and 64 MiB. The collector retains immutable byte
references in memory so disk writes do not interrupt camera drainage. It also
retains up to 128 existing stream-health events and the negotiated mode in
`index.json`. These are original backend-delivered frames before decoding,
including invalid radiometry; this is not a USB packet trace. Scene images remain
local ignored diagnostic artifacts. Run this live check separately from pytest.

## Markup compositor (no external resources)

For display-only Markup performance and PNG previews without camera access:

```sh
.venv/bin/python experiments/experiment_markup.py --frames 100 --output diagnostics/markup
```

This measures the actual main-process compositor on a synthetic 768×576 frame
with 1, 10 and 128 annotations plus a large live table with changing readouts,
reporting cold, warm median and p95 milliseconds. Optional PNG output includes
`markup-0.png` for the live table.
It uses existing dependencies and never reads user preferences or opens USB.

## Thermal Surface renderer (no external resources)

```sh
.venv/bin/python experiments/experiment_thermal_surface.py --frames 200 --output diagnostics/thermal-surface
```

Runs the actual portable Custom node on a synthetic heat image at three quality
settings, reporting first-frame and warm median/p95 timings and optionally writing
PNG previews. It uses existing NumPy/OpenCV dependencies, performs no downloads,
and does not open hardware or modify preferences. See the
[Thermal Surface README](../examples/custom_nodes/Thermal%20Surface/README.md).

## Optional visual ONNX weights

```sh
.venv/bin/python experiments/experiment_onnx_visual.py --models realesr-general-x4v3
.venv/bin/python experiments/experiment_onnx_visual.py
# On macOS, also check V1 Core ML initialization:
.venv/bin/python experiments/experiment_onnx_visual.py --coreml
```

Install the desired optional models separately using the README's model installer
or documented conversion commands. The experiment verifies installed checksums
before inference and reports missing weights as skipped. It checks scaled output
geometry and compares V1 dynamic/fixed dimension CPU outputs when those weights
are present. It does not download models or export weights. Core ML requires its
ONNX Runtime provider and may create compiler caches; compilation restrictions
are reported as failures. A failed check returns a nonzero exit status.

Normal application tests retain fake sessions and isolated temporary model files
to cover pipeline behavior independently of the user's model cache.

## Head Pose

Preserve the upstream checkout and prepare optional pinned external weights separately:

```sh
git submodule update --init -- examples/custom_nodes/head-pose/upstream
uv run --python .venv/bin/python --script tools/prepare_head_pose.py
.venv/bin/python experiments/experiment_head_pose.py \
  --images /path/to/head-image.png --backends cpu coreml --pose-backend cpu
```

Preparation uses isolated ONNX/certificate dependencies and writes into ignored
`diagnostics/head-pose/models`. Preparation is optional for viewer use: the Custom
node's JSON detector dropdown and empty local overrides download audited pinned
weights automatically. The experiment measures real Custom worker cold/warm
timings, writes annotated stills and predictions, and exports portable local-weight
presets under `diagnostics/head-pose/results`. It does not acquire a camera or alter
preferences. CPU-only hosts can use `--backends cpu`; unavailable GPU requests use
the normal fallback, and the report records the registered providers. The
[Head Pose README](../examples/custom_nodes/head-pose/README.md) documents the
verified model schemas, measured backends, update workflow and thermal limitations.

## First-use denoiser preparation

```sh
.venv/bin/python experiments/experiment_model_preparation.py dncnn-25 ffdnet-gray --model-dir /tmp/based-thermals-denoiser-check
```

This explicitly downloads isolated conversion dependencies and official weights,
checks conversion parity, verifies offline cache reuse, and runs same-size CPU
inference. Use a separate model directory as shown to preserve viewer data. It
requires uv and internet access; normal pytest never performs conversion or downloads.
