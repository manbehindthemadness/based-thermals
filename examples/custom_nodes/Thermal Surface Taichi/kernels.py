"""
Rasterize the existing painter-ordered mesh without windows or native UI resources.

Taichi 1.7.4's AST transformer rejects explicit None return annotations on void kernels.
"""

import importlib
from typing import TYPE_CHECKING

ti = importlib.import_module("taichi")

if TYPE_CHECKING:
    import numpy as np

    Triangles = np.ndarray
    Owners = np.ndarray
    Colors = np.ndarray
    Pixels = np.ndarray
else:
    Triangles = ti.types.ndarray(dtype=ti.f32, ndim=3)
    Owners = ti.types.ndarray(dtype=ti.i32, ndim=2)
    Colors = ti.types.ndarray(dtype=ti.u8, ndim=2)
    Pixels = ti.types.ndarray(dtype=ti.u8, ndim=3)


@ti.kernel
def rasterize(triangles: Triangles, owners: Owners, wireframe: ti.i32, scale: ti.i32):
    """
    Resolve triangle overlap by the original painter rank using deterministic integer atomics.
    """
    for rank in range(triangles.shape[0]):
        a = ti.Vector([ti.cast(triangles[rank, 0, k], ti.f32) for k in ti.static(range(2))])
        b = ti.Vector([ti.cast(triangles[rank, 1, k], ti.f32) for k in ti.static(range(2))])
        c = ti.Vector([ti.cast(triangles[rank, 2, k], ti.f32) for k in ti.static(range(2))])
        a = a * scale + (scale - 1) * 0.5
        b = b * scale + (scale - 1) * 0.5
        c = c * scale + (scale - 1) * 0.5
        area = (b - a).cross(c - a)
        if ti.abs(area) > 0.01:
            low_x = ti.max(0, ti.cast(ti.min(a.x, b.x, c.x), ti.i32))
            high_x = ti.min(owners.shape[1], ti.cast(ti.max(a.x, b.x, c.x), ti.i32) + 1)
            low_y = ti.max(0, ti.cast(ti.min(a.y, b.y, c.y), ti.i32))
            high_y = ti.min(owners.shape[0], ti.cast(ti.max(a.y, b.y, c.y), ti.i32) + 1)
            for y in range(low_y, high_y):
                for x in range(low_x, high_x):
                    p = ti.Vector([ti.cast(x, ti.f32), ti.cast(y, ti.f32)])
                    e0, e1, e2 = (b - a).cross(p - a), (c - b).cross(p - b), (a - c).cross(p - c)
                    if e0 * area >= 0 and e1 * area >= 0 and e2 * area >= 0:
                        edge = 0
                        if wireframe:
                            distance = ti.min(ti.abs(e0) / ti.max((b - a).norm(), 1.0),
                                              ti.abs(e1) / ti.max((c - b).norm(), 1.0),
                                              ti.abs(e2) / ti.max((a - c).norm(), 1.0))
                            edge = ti.cast(distance <= 0.6 * scale, ti.i32)
                        ti.atomic_max(owners[y, x], rank * 2 + edge)


@ti.kernel
def resolve(owners: Owners, colors: Colors, floor: Pixels, output: Pixels, scale: ti.i32):
    """
    Produce the BGR host image, retaining the existing cached floor outside the mesh.
    """
    for y, x in ti.ndrange(output.shape[0], output.shape[1]):
        for channel in ti.static(range(3)):
            total = 0
            for sy, sx in ti.ndrange(scale, scale):
                winner = owners[y * scale + sy, x * scale + sx]
                value = floor[y, x, channel]
                if winner >= 0:
                    value = colors[winner // 2, channel]
                    if winner % 2:
                        value = ti.static((35, 33, 30))[channel]
                total += ti.cast(value, ti.i32)
            output[y, x, channel] = ti.cast(ti.floor(total / (scale * scale) + 0.5), ti.u8)
