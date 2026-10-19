# Taichi ARM64 build notes

## Sources and build selection

Viewer startup installs 1.7.4 on platforms with published wheels. Taichi remains
outside `pyproject.toml` and `uv.lock`. Linux ARM64 has no published wheel; build
an optional CPU + Vulkan wheel and register it in the separate application cache.

Primary sources checked on 2026-10-19:

- [Taichi developer installation](https://docs.taichi-lang.org/docs/dev_install):
  matching LLVM 15, recursive submodules, wheel builds and explicit Vulkan enablement.
- [Pinned source](https://github.com/taichi-dev/taichi/tree/b4b956fdaa67fd3e198527fd607053ecb9bf4aa2).
- [Runtime bitcode build](https://github.com/taichi-dev/taichi/blob/b4b956fdaa67fd3e198527fd607053ecb9bf4aa2/taichi/runtime/llvm/runtime_module/CMakeLists.txt).
- [Clang ARM atomics option](https://releases.llvm.org/17.0.1/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-moutline-atomics):
  `-mno-outline-atomics` prevents calls to external atomic helpers.
- [Upstream build helper](https://github.com/taichi-dev/taichi/blob/b4b956fdaa67fd3e198527fd607053ecb9bf4aa2/.github/workflows/scripts/ti_build/llvm.py):
  Linux's prebuilt LLVM selection does not check architecture, so use controlled
  `setup.py bdist_wheel` instead.
- [Published wheels](https://pypi.org/project/taichi/1.7.4/#files).

## Prerequisites and toolchain

Use native Linux ARM64 CPython 3.12/3.13 in the project venv, `uv`, `git`, C/C++
headers, matching LLVM/Clang 15 and X11 development headers. Ubuntu's typical
prerequisites include `build-essential python3-dev libffi-dev libtinfo-dev libz3-dev
libxml2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`.
Python headers must match the interpreter. Report specific missing packages;
do not change drivers or unrelated OS settings.

The helper prefers system `/usr/lib/llvm-15`. Otherwise, on Ubuntu 24.04 it downloads
and extracts native LLVM/Clang 15 packages under diagnostics without sudo. Exact
APT candidates are recorded in `llvm-packages.txt`; it does not refresh indexes or
install packages system-wide. Observed Ubuntu version: `1:15.0.7-14build3`.
For other distributions use `--llvm-dir` with a prefix containing `bin/clang`,
`bin/clang++`, `bin/llvm-config` and `lib/cmake/llvm`. LLVM 18 is not an API/bitcode
replacement for 15. Build-only loader paths never enter runtime verification or
user startup files. The wheel must work without the extracted toolchain loaded.

Vulkan headers/compiler libraries are pinned submodules; no downloaded Vulkan SDK,
`libvulkan-dev` or CUDA toolkit was needed. GGUI is enabled by upstream Vulkan,
which explains the X11 prerequisites. Compilation defaults to at most eight jobs;
reduce to `--jobs 2` on small systems and allow several GB of disk space.

## CPU fix and preserved fallback

Unmodified LLVM 15 runtime bitcode called `__aarch64_ldadd4_acq_rel`,
`__aarch64_ldadd8_acq_rel`, `__aarch64_swp4_acq_rel` and `__aarch64_swp8_acq_rel`.
Taichi's JIT could not resolve those external libgcc atomic helpers; initialization
failed and teardown raised SIGBUS. The early Vulkan-only workaround avoided that
failure by disabling LLVM, but also made **Taichi CPU** unavailable.

The delivered [patch](../assets/arm64-no-outline-atomics.patch) adds
`-mno-outline-atomics` only when compiling Linux `arm64` runtime bitcode. This keeps
LLVM CPU support while eliminating the unresolvable helper calls. Source commit
and patch SHA256 are distinct inputs in `build-identity.json` and `manifest.json`.
Reapplication is idempotent; unrelated tracked source edits are refused.

CPU numerical verification is mandatory and stops publication on failure. Vulkan
verification runs separately; missing drivers, software devices, crashes or timeouts
do not discard a working CPU build. Registration uses `--arch cpu`, so cache reuse
never depends on the GPU remaining available. The node still uses OpenCV CPU
rendering when Taichi or the selected accelerator fails, preserving saved selection.
An old Vulkan-only package is checked before CPU/CUDA initialization to give an
upgrade instruction without triggering Taichi's native LLVM error.

CUDA remains off: native ARM Clang 15 failed at `runtime.cpp:1160` with
`invalid input constraint 'l' in asm`. Delivered flags enable LLVM and Vulkan;
CUDA, OpenGL, Metal, C API and native tests are off. This does not promise CUDA.

## Other reproducibility details

- Tag `v1.7.4` is commit `b4b956fdaa67fd3e198527fd607053ecb9bf4aa2`.
- GitLab's Eigen git endpoint stalled. The helper uses the historical official
  `eigenteam/eigen-git-mirror` with exact pinned submodule commit
  `0fd6b4f71dd85b2009ee4d1aeb296e2c11fc9d68`, never mirror HEAD.
- Await completed submodule checkout. Missing `spdlog/common.h` indicates an
  incomplete checkout, not a pip dependency.
- Build dependencies are pinned in their own venv. Scikit-build needs CMake below 4.
- A local `linux_aarch64` wheel is not an audited manylinux distribution. Do not
  promise other libc/ABI/driver compatibility. Upstream license files stay in the wheel.

## Artifacts and recovery

Default folder: `diagnostics/taichi-cpu-vulkan-build/`.

- `source/dist/taichi-1.7.4-cp312-cp312-linux_aarch64.whl` (CPython 3.12).
- `manifest.json`: commit, patch hash, wheel hash, verified architectures and cache path.
- `build.log`, `verify-{cpu,vulkan}.{log,json}`, `installed-runtime.{log,json}`.
- `llvm-packages.txt`, `submodules.txt`, `build-identity.json` retain reproduction inputs.
- `surface-numpy/` and `surface-taichi/` contain camera-free application reports.

Normal `uv sync` leaves the runtime cache intact. Register a retained CPU-capable
wheel without recompilation with `python -m based_thermals.runtime_setup.taichi
--wheel WHEEL --arch cpu`. Restart the viewer to replace an imported native runtime.
Remove its `taichi-v1-*` cache family with viewers closed to uninstall. ARM Linux
will not download or compile a replacement automatically.

## Measured scope (2026-10-19)

Ubuntu 24.04.4, Linux 6.17.0-1029-nvidia, aarch64/glibc 2.39; CPython 3.12.3,
LLVM/Clang 15.0.7; NVIDIA GB10 driver 580.173.02 / Vulkan 1.4.312.
The patched build passes a 4,096-value numerical kernel on CPU and Vulkan, and the
Thermal Surface experiment with compiled geometry on both backends. The independent
OpenCV reference, control changes, concurrent nodes and reload checks pass. A forced
missing Vulkan ICD still passes CPU verification and selects CPU-only registration.
These correctness checks do not establish a speedup or physical-camera compatibility.
Other ARM GPUs, distributions and Python 3.13 remain unverified. Exact build hashes
and final runtime paths are in the local manifest rather than portable instructions.
