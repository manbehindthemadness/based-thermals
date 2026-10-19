---
name: build-taichi
description: Compile, verify and install Taichi with CPU fallback and Vulkan on Linux ARM64 when PyPI wheels are unavailable. Use for Taichi source installation; general GPU inference integration and image filters have separate workflows.
---

# Build Taichi for Linux ARM64

Build pinned Taichi 1.7.4 with LLVM CPU and Vulkan support, then register it in the
application runtime cache outside `.venv`. A passing CPU kernel is required. Vulkan
hardware is optional: missing drivers or a failed GPU probe must leave CPU rendering
available. Keep Taichi out of `pyproject.toml` and `uv.lock`; Linux ARM compilation
is opt-in, while viewer startup installs published wheels on supported wheel hosts.

Read [build notes](references/build-notes.md) for prerequisites, the ARM atomic fix,
provenance and measured results. Inspect repository instructions and changes first.
Use the configured project interpreter; without IDE Python tools, verify
`.venv/bin/python` and `uv python find` explicitly.

## Build and register

From the project root, with a native Linux ARM64 CPython 3.12 virtual environment
and the native development prerequisites:

```sh
.venv/bin/python .agents/skills/build-taichi/scripts/build_taichi.py
```

The helper uses LLVM/Clang 15. On Ubuntu 24.04 it can download and extract the
native toolchain into diagnostics without sudo; it changes no system packages.
On another distribution install a matching toolchain or supply its prefix with
`--llvm-dir /path/to/llvm-15` (containing `bin` and `lib/cmake/llvm`). Do not substitute
LLVM 18 for 15: Taichi's JIT/runtime bitcode must use matching LLVM APIs.

Sources, toolchain, wheel, manifest and logs stay under ignored
`diagnostics/taichi-cpu-vulkan-build/`. Track `build.log` during compilation.

- Use `--jobs 2` on memory-limited hosts; the default is at most eight jobs.
- Use `--no-install` to stop after staged verification without registering the wheel.
- Use `--project`, `--python` and `--build-dir` for another checkout/interpreter or
  build location. Use a fresh directory when changing ABI or compiler settings.
- CPython 3.12 is measured; verify Python 3.13 on its own target before claiming it.

The helper applies the recorded [ARM atomics patch](assets/arm64-no-outline-atomics.patch)
only to Taichi's Linux ARM64 runtime-bitcode compilation. It disables outlined
atomic helper calls that the JIT cannot resolve. The source commit and patch hash
are recorded separately. Do not remove LLVM to make a CPU failure disappear.
CUDA remains disabled because of a separate upstream ARM compilation failure.

A fresh verification venv runs a numerical CPU kernel before any GPU experiment.
CPU failure stops registration. Vulkan failure is recorded and retains CPU support;
missing, malformed and non-boolean verification replies also count as failures.
common software Vulkan devices do not count as verified GPU execution. Registration
uses `--arch cpu`, so startup/cache reuse also works without a Vulkan driver.
The node retains its independent OpenCV CPU fallback if Taichi is unavailable.

## Verify the application

```sh
.venv/bin/python experiments/experiment_taichi_surface.py \
  --backends opencv cpu vulkan --frames 3 --details 32 --geometry NumPy \
  --timeout 180 --output diagnostics/taichi-cpu-vulkan-build/surface-numpy
.venv/bin/python experiments/experiment_taichi_surface.py \
  --backends cpu vulkan --frames 3 --details 32 --geometry Taichi \
  --timeout 180 --output diagnostics/taichi-cpu-vulkan-build/surface-taichi
.venv/bin/python -m pytest tests/test_taichi_build_skill.py \
  tests/test_taichi_surface_package.py tests/test_taichi_runtime_setup.py tests/test_thermal_surface.py
```

Omit `vulkan` from experiments on CPU-only machines. Check actual architectures,
finite numerical output, CPU/reference agreement and logs. A small kernel does
not establish camera compatibility or a speedup. Keep failed native process logs.

Registered builds survive ordinary exact `uv sync`. Restart an existing viewer to
load a replacement runtime; switching a loaded native library in-process is unsafe.
**Taichi CPU** is the node's portable default, **Taichi Vulkan** selects the GPU,
and **OpenCV** remains a CPU option. Saved documents embed package code: reopen
this module folder and save if an older embedded version needs its preflight fix.

Register a retained CPU-capable wheel without rebuilding:

```sh
.venv/bin/python -m based_thermals.runtime_setup.taichi \
  --wheel diagnostics/taichi-cpu-vulkan-build/source/dist/taichi-1.7.4-cp312-cp312-linux_aarch64.whl \
  --arch cpu
```

Use the actual filename for another Python ABI. Failed replacements preserve the
previous cache generation. `BASED_THERMALS_TAICHI_SETUP=0` disables Taichi cache
activation/setup; `BASED_THERMALS_RUNTIME_SETUP=0` disables all runtime setup.
See [runtime setup](../../../docs/taichi-runtime-setup.md) for controls and recovery.

Report wheel path/hash, source commit plus patch, Python/LLVM versions, tested
CPU/GPU results and limitations. Keep binaries local unless publication is requested.
