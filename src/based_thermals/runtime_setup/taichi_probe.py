"""
Verify optional Taichi kernels in a disposable process, without opening a camera.
"""

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


# The standalone build verifier intentionally shares this independent numerical kernel.
# noinspection DuplicatedCode
def verify(arch: str) -> dict[str, Any]:
    """
    Require the pinned package and completed numerical output on the requested backend.
    """
    os.environ["TI_SKIP_VERSION_CHECK"] = "ON"
    import numpy as np
    import taichi as ti

    if tuple(ti.__version__) != (1, 7, 4):
        raise ValueError("Expected Taichi 1.7.4")
    requested = getattr(ti, arch)
    ti.init(arch=requested, enable_fallback=False, offline_cache=False, log_level=ti.TRACE)
    ti.set_logging_level(ti.INFO)
    if ti.lang.impl.current_cfg().arch != requested:
        raise ValueError("Taichi selected a different backend")

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
    verified = {"available": True, "arch": arch, "version": "1.7.4", "module": ti.__file__}
    return verified


if __name__ == "__main__":
    destination = Path(sys.argv[2])
    try:
        result = verify(sys.argv[1])
    except Exception as exc:  # noqa: BLE001 - report optional native initialization failures
        result = {"available": False, "reason": str(exc)}
    destination.write_text(json.dumps(result), encoding="utf-8")
