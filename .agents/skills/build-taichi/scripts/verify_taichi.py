"""
Run a real kernel with fallback disabled; execute from a file for Taichi's AST reader.
"""

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("TI_SKIP_VERSION_CHECK", "ON")

import numpy as np
import taichi as ti


# This verifier runs without the application installed; retain the matching numerical probe.
# noinspection DuplicatedCode
def main() -> None:
    """
    Verify an exact backend numerically and publish a standalone build-check response.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("vulkan", "cpu", "cuda"), default="vulkan")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arch = getattr(ti, args.arch)
    ti.init(arch=arch, enable_fallback=False, offline_cache=False, log_level=ti.TRACE)
    ti.set_logging_level(ti.INFO)
    actual = ti.lang.impl.current_cfg().arch
    if actual != arch:
        raise RuntimeError(f"Requested {arch}, got {actual}")

    if TYPE_CHECKING:
        FloatArray = np.ndarray
    else:
        # Taichi consumes this runtime type when decorating the kernel.
        # noinspection PyUnusedLocal
        FloatArray = ti.types.ndarray(dtype=ti.f32, ndim=1)

    # Taichi 1.7.4 rejects -> None on void kernels; these are runtime DSL annotations.
    @ti.kernel
    def transform(inputs: FloatArray, outputs: FloatArray):
        """
        Exercise native arithmetic over a host array with independently checked output.
        """
        for i in inputs:
            outputs[i] = inputs[i] * 2.0 + 1.0

    x = np.arange(4096, dtype=np.float32)
    y = np.empty_like(x)
    transform(x, y)
    ti.sync()
    np.testing.assert_array_equal(y, x * 2 + 1)
    args.output.write_text(json.dumps({
        "ok": True, "requested_arch": args.arch, "actual_arch": str(actual),
        "version": list(ti.__version__), "module": ti.__file__, "elements": len(x),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
