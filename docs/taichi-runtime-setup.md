# Optional Taichi runtime

[Setup](setup.md) · [Thermal Surface](../examples/custom_nodes/Thermal%20Surface%20Taichi/README.md#installation)

Taichi is optional and is not a project dependency in `pyproject.toml` or `uv.lock`.
Normal viewer startup prepares it before camera construction or Custom processing.
`uv sync` installs the application dependencies; launching the viewer separately
installs or reuses Taichi in the application runtime cache, outside `.venv`.
Ordinary exact synchronization cannot remove that cache.

| Host | Startup behavior |
| --- | --- |
| Apple Silicon macOS, CPython 3.12–3.13 | Install published `taichi==1.7.4` wheel if no verified cache exists |
| x86-64 Linux, CPython 3.12–3.13 | Same automatic wheel setup |
| ARM64 Linux, CPython 3.12–3.13 | Activate a previously registered source build; never automatically compile or download Taichi |
| Other hosts/ABIs, including Python 3.14+ | Keep OpenCV fallback |

This selection follows [Taichi 1.7.4's published wheels](https://pypi.org/project/taichi/1.7.4/#files),
checked on 2026-10-19; this release also lacks Intel macOS wheels. Availability of
a package does not guarantee a working GPU backend. Published wheels are verified
with a numerical CPU kernel; the node initializes the requested GPU backend when
processing. Native backend failures retain its existing OpenCV fallback.

The cache uses the same base as ONNX Runtime: `~/Library/Caches/based-thermals/runtimes`
on macOS, or `$XDG_CACHE_HOME/based-thermals/runtimes` (default
`~/.cache/based-thermals/runtimes`) on Linux. `BASED_THERMALS_RUNTIME_DIR` overrides
the base. Taichi has separate families keyed by version, Python ABI, OS and process
architecture, with verified generations and a state file. Its selected path is
passed to workers through `PYTHONPATH` and `BASED_THERMALS_TAICHI_PATH`, preserving
the independent ONNX Runtime selection.

Setup uses `uv` on PATH or beside the running interpreter. It installs Taichi and
its Python helpers in staging, using the project's existing NumPy. A disposable
process must import the staged package and produce correct numerical output before
the generation is published. Native crashes, malformed responses and 45-second
probe timeouts fail softly. Wheel downloads have ten-minute timeouts. Failed setup
has a five-minute retry cooldown; process locking prevents concurrent installations.
Interrupted staging directories are cleaned on the next attempt. No driver, system
package or administrator setting is changed. Read-only/offline failures keep OpenCV.

To prepare or retry only Taichi without opening a camera:

```sh
.venv/bin/python -m based_thermals.runtime_setup.taichi --retry
```

To disable Taichi startup setup and cache activation:

```sh
BASED_THERMALS_TAICHI_SETUP=0 .venv/bin/based-thermals-desktop
```

`BASED_THERMALS_RUNTIME_SETUP=0` disables both Taichi and ONNX Runtime startup
setup/activation. Help and USB diagnostic requests do not provision packages.
For embedded use, call `based_thermals.runtime_setup.ensure_runtime()` before
native imports or worker startup. Setup does not replace an already imported Taichi.
These commands select packages in their own process; they do not alter the parent shell.

## Optional ARM64 Linux source build

Use the [build-taichi skill](../.agents/skills/build-taichi/SKILL.md) or its helper:

```sh
.venv/bin/python .agents/skills/build-taichi/scripts/build_taichi.py
```

The helper builds a CPU + Vulkan wheel, requires a passing CPU kernel, and registers
it in the runtime cache. A failed GPU check retains the working CPU runtime.
It does not install Taichi into `.venv`. Subsequent viewer launches discover
it automatically; exact `uv sync` and recreating `.venv` with the same Python ABI
leave the cache intact. Python ABI changes need a matching source build.

Register an already compiled wheel without rebuilding:

```sh
.venv/bin/python -m based_thermals.runtime_setup.taichi \
  --wheel diagnostics/taichi-cpu-vulkan-build/source/dist/taichi-1.7.4-cp312-cp312-linux_aarch64.whl \
  --arch cpu
```

Registration and startup verify CPU execution so GPU/driver availability cannot
disable CPU compatibility. Choose **Taichi CPU** (the node default) or **Taichi Vulkan**
after restarting the viewer; OpenCV remains an independent CPU fallback. This build
omits CUDA. Older Vulkan-only builds must be rebuilt to enable Taichi CPU; selecting
CPU in an old build now gives an upgrade instruction before native initialization.
Existing camera-free Taichi surface
experiments activate cached runtimes without installing packages, and retain their
`--runtime` override for isolated experiments.

To remove Taichi, close viewers and remove its `taichi-v1-*` cache family. On ARM
Linux it remains absent until explicitly registered again. On a wheel host, disable
Taichi setup as above to prevent its installation on the next launch.
