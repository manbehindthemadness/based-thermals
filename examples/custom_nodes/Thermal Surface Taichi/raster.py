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

from .geometry import Geometry
from .native_mesh import NativeMesh
from .settings import Settings

LOG = logging.getLogger(__name__)
EXECUTIONS = {"Taichi CPU": "cpu", "Taichi Metal": "metal",
              "Taichi Vulkan": "vulkan", "Taichi CUDA": "cuda"}
_REGISTRY = "_based_thermals_surface_raster_v1"


def kernel_digest() -> str:
    """
    Detect source updates without retaining a new kernel library for every reload.
    """
    source = b"".join(Path(__file__).with_name(name).read_bytes()
                      for name in ("kernels.py", "mesh_kernels.py"))
    return sha256(source).hexdigest()


def service(execution: str) -> Any:
    """
    Share one bounded kernel library and lock across separately imported node namespaces.

    Never reset an existing runtime: doing so would invalidate other nodes' allocations.
    """
    os.environ.setdefault("TI_SKIP_VERSION_CHECK", "ON")
    ti = importlib.import_module("taichi")
    # Taichi 1.7.4 exposes LLVM build support only through its native extension.
    # noinspection PyProtectedMember
    if (EXECUTIONS[execution] in {"cpu", "cuda"}
            and ti._lib.core.get_llvm_target_support() == "targets unsupported"):
        raise RuntimeError(
            f"{execution} needs an LLVM-enabled Taichi build. Rebuild with the build-taichi "
            "helper for CPU + Vulkan support, or select OpenCV for CPU rendering."
        )
    state: Any = ti.__dict__.setdefault(_REGISTRY, SimpleNamespace(lock=RLock(), library=None))
    with state.lock:
        requested = getattr(ti, EXECUTIONS[execution])
        runtime: Any = ti.lang.impl.get_runtime()
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
        self.mesh: NativeMesh | None = None

    def render(self, floor: np.ndarray, triangles: np.ndarray,
               colors: np.ndarray, wireframe: bool, sample_scale: int = 1) -> np.ndarray:
        """
        Include host/device copies and completed readback in the synchronous callback duration.
        """
        with self.state.lock:
            output = self._render(floor, np.ascontiguousarray(triangles, np.float32),
                                  np.ascontiguousarray(colors, np.uint8), wireframe, sample_scale)
        return output

    def render_mesh(self, heat: np.ndarray, geometry: Geometry, settings: Settings,
                    palette: np.ndarray, sample_scale: int) -> np.ndarray:
        """
        Run compiled geometry and rasterization under the shared runtime lock with node-owned buffers.
        """
        ti = importlib.import_module("taichi")
        with self.state.lock:
            if ti.lang.impl.get_runtime().prog is not self.state.program:
                raise RuntimeError("Taichi runtime was reset outside this node")
            library = getattr(self.state, "mesh_library", None)
            if library is None:
                library = importlib.import_module(".mesh_kernels", __package__)
                self.state.mesh_library = library
            mesh = self.mesh
            if mesh is None or mesh.geometry is not geometry:
                mesh = NativeMesh(ti, geometry, settings)
                self.mesh = mesh
            triangles, colors = mesh.prepare(library, heat, palette, settings.antialiasing == "Off")
            output = self._render(geometry.floor, triangles, colors, settings.wireframe, sample_scale)
        return output

    def _render(self, floor: np.ndarray, triangles: Any, colors: Any,
                wireframe: bool, sample_scale: int) -> np.ndarray:
        """
        Rasterize host or resident mesh arrays while the caller holds the runtime lock.
        """
        ti = importlib.import_module("taichi")
        if ti.lang.impl.get_runtime().prog is not self.state.program:
            raise RuntimeError("Taichi runtime was reset outside this node")
        height, width = floor.shape[:2]
        sample_shape = height * sample_scale, width * sample_scale
        if self.shape != sample_shape:
            self.owners = ti.ndarray(dtype=ti.i32, shape=sample_shape)
            self.shape = sample_shape
        self.owners.fill(-1)
        output = np.empty_like(floor)
        self.state.library.rasterize(triangles, self.owners, int(wireframe), sample_scale)
        self.state.library.resolve(self.owners, colors, np.ascontiguousarray(floor), output, sample_scale)
        ti.sync()
        return output
