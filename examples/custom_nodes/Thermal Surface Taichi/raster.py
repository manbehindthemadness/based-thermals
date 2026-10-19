"""
Own optional Taichi initialization and per-node buffers entirely within this Custom package.
"""

import importlib
import logging
import os
from hashlib import sha256
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)
EXECUTIONS = {"Taichi CPU": "cpu", "Taichi Metal": "metal",
              "Taichi Vulkan": "vulkan", "Taichi CUDA": "cuda"}
_REGISTRY = "_based_thermals_surface_raster_v1"


def kernel_digest() -> str:
    """
    Detect source updates without retaining a new kernel library for every reload.
    """
    source = Path(__file__).with_name("kernels.py").read_bytes()
    return sha256(source).hexdigest()


def service(execution: str) -> Any:
    """
    Share one bounded kernel library and lock across separately imported node namespaces.

    Never reset an existing runtime: doing so would invalidate other nodes' allocations.
    """
    os.environ.setdefault("TI_SKIP_VERSION_CHECK", "ON")
    ti = importlib.import_module("taichi")
    state = ti.__dict__.setdefault(_REGISTRY, SimpleNamespace(lock=RLock(), library=None))
    with state.lock:
        requested = getattr(ti, EXECUTIONS[execution])
        runtime = ti.lang.impl.get_runtime()
        digest = kernel_digest()
        if state.library is None:
            if runtime.prog is None:
                ti.init(arch=requested, enable_fallback=False, offline_cache=False,
                        log_level=ti.ERROR)
            elif ti.lang.impl.current_cfg().arch != requested:
                raise RuntimeError("Taichi already uses another backend; restart to change it")
            state.library = importlib.import_module(".kernels", __package__)
            state.program = ti.lang.impl.get_runtime().prog
            state.arch = requested
            state.digest = digest
        if state.digest != digest:
            raise RuntimeError("Taichi kernel source changed; restart to load the new kernels")
        if ti.lang.impl.get_runtime().prog is not state.program or state.arch != requested:
            raise RuntimeError("Taichi runtime changed or uses another backend; restart to change it")
    return state


class Rasterizer:
    """
    Retain only this node's current owner buffer; kernels hold no node or frame references.
    """

    def __init__(self, execution: str) -> None:
        """
        Resolve a compatible shared runtime without importing Taichi during document browsing.
        """
        self.state = service(execution)
        self.owners: Any = None
        self.shape: tuple[int, int] | None = None

    def render(self, floor: np.ndarray, triangles: np.ndarray,
               colors: np.ndarray, wireframe: bool, sample_scale: int = 1) -> np.ndarray:
        """
        Include host/device copies and completed readback in the synchronous callback duration.
        """
        ti = importlib.import_module("taichi")
        height, width = floor.shape[:2]
        sample_shape = height * sample_scale, width * sample_scale
        with self.state.lock:
            if ti.lang.impl.get_runtime().prog is not self.state.program:
                raise RuntimeError("Taichi runtime was reset outside this node")
            if self.shape != sample_shape:
                self.owners = ti.ndarray(dtype=ti.i32, shape=sample_shape)
                self.shape = sample_shape
            self.owners.fill(-1)
            output = np.empty_like(floor)
            self.state.library.rasterize(np.ascontiguousarray(triangles, np.float32),
                                         self.owners, int(wireframe), sample_scale)
            self.state.library.resolve(self.owners, np.ascontiguousarray(colors, np.uint8),
                                       np.ascontiguousarray(floor), output, sample_scale)
            ti.sync()
        return output
