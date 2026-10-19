# Use the supplied Codex skills

[Documentation home](../README.md) · [Custom pipeline nodes](custom-nodes.md)

The repository includes four skills under `.agents/skills/`. They are instructions
and supporting resources for a coding agent working on this project. They are
separate from runtime Custom nodes: you do not load a SKILL.md into the viewer,
and you do not need Codex to use Capture, Imaging or Measurements.

## Access them

1. Clone/open this repository, keeping the checked-in `.agents/skills/` directory.
2. Start a Codex session with this checkout as its working project, such as a
   Codex CLI/desktop session or your IDE's Codex integration.
3. Name the skill explicitly in your request using `$skill-name`, followed by
   the goal, target hardware/model and any constraints.
4. Review the proposed changes and reported evidence before adopting the result.

Codex discovers repository skills from `.agents/skills/` between its current
directory and the repository root. It can also select a relevant skill automatically.
If a new skill is missing, verify the working directory and start a fresh/restarted
Codex session. Discovery/settings vary by client; see the official
[local skill documentation](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).
Using these repo-scoped skills does not require moving them into a global directory
or installing them as a plugin.

## Create a Custom node

[$create-custom-node](../.agents/skills/create-custom-node/SKILL.md) creates/adapts
folder-based Python/JSON packages for the software Custom node, including controls,
model resolution, portable exports and focused verification.

Example request:

```text
Use $create-custom-node to create an adjustable local-contrast node in
examples/custom_nodes/my_contrast. Add an amount control and preserve image size.
```

Provide the desired effect, controls, expected input/output, model source if any,
and destination. Load the resulting trusted package through Imaging's Custom node.
See [package instructions](custom-nodes.md) for the runtime contract. Imported
package code runs with application permissions; skill files themselves are not
processing plugins.

## Discover or update a GPU backend

[$integrate-gpu-backend](../.agents/skills/integrate-gpu-backend/SKILL.md) researches
runtime/driver/GPU compatibility and implements evidenced support across startup
setup, workers, execution controls and Custom devices while retaining CPU fallback.

```text
Use $integrate-gpu-backend to investigate this CUDA version and GPU. First identify
the compatible ONNX Runtime build, then implement and verify support without
changing the system driver or breaking the existing backends.
```

```text
Use $integrate-gpu-backend to check CoreML acceleration on this Apple M5 host.
Measure real operator execution and pipeline performance before changing defaults.
```

Provide OS, GPU, Python/architecture, driver/runtime versions, error logs, and
whether the target machine is available. A new chip name is not proof that a new
backend/wheel is needed. The skill distinguishes published compatibility, package
resolution, actual GPU execution and useful speedup; unavailable hardware remains
explicitly unverified.

## Build Taichi on Linux ARM64

[$build-taichi](../.agents/skills/build-taichi/SKILL.md) compiles the pinned Taichi
release with Vulkan support, verifies a real GPU kernel in staging, and installs
a reusable wheel into the application's separate runtime cache. It includes LLVM
CPU support for compatibility and Vulkan for GPU rendering. On Ubuntu 24.04 the
helper can extract LLVM/Clang 15 locally without sudo; a CUDA toolkit is unnecessary.

```text
Use $build-taichi to build and install Taichi with Vulkan on this ARM64 Linux host.
Verify the Thermal Surface Taichi node and retain the wheel for offline reinstall.
```

To run the helper directly from the project root:

```sh
.venv/bin/python .agents/skills/build-taichi/scripts/build_taichi.py
```

See the skill's [build notes](../.agents/skills/build-taichi/references/build-notes.md)
for prerequisites and limitations. The registered runtime survives ordinary `uv sync`.

## Research another thermal camera

[$reverse-engineer-thermal-camera](../.agents/skills/reverse-engineer-thermal-camera/SKILL.md)
investigates transport, radiometry, telemetry and reversible controls, then integrates
verified capabilities. It includes the Duo case study and offline evidence helpers.

```text
Use $reverse-engineer-thermal-camera to investigate this camera's capture format
and native temperature data. Start with descriptors and vendor sources; integrate
a read-only backend before attempting controls.
```

Provide model/VID/PID, firmware, OS, vendor SDK/app sources and saved captures if
available. State whether another application currently owns the camera. The skill
requires recoverable, evidenced trials and excludes firmware/flash/NVM/factory-reset
experiments from its normal workflow. Loading the skill alone does not run a probe.

## If your coding assistant does not support skill discovery

Open the linked SKILL.md and its relevant references, or ask the agent explicitly:

```text
Read .agents/skills/integrate-gpu-backend/SKILL.md and use its workflow for this task.
```

This makes the instructions accessible; it does not supply missing shell, IDE,
network or hardware tools. These project skills use the current repository rules
and may rely on this IDE's `python-tools`/environment integration. A client without
those tools needs an equivalent configured environment workflow, with missing
verification reported. Do not assume hardware support was proven by an offline
code change, or treat a skill request as permission for unrelated system changes.
