"""
Project and shade the height mesh in the selected Taichi runtime.

Void kernels omit None return annotations for Taichi 1.7.4 compatibility.
"""

import importlib
from typing import TYPE_CHECKING

ti = importlib.import_module("taichi")

if TYPE_CHECKING:
    import numpy as np

    Floats = Points = Vertices = Triangles = Depths = np.ndarray
    Indices = Order = Colors = np.ndarray
else:
    Floats = Depths = ti.types.ndarray(dtype=ti.f32, ndim=1)
    Points = Vertices = ti.types.ndarray(dtype=ti.f32, ndim=2)
    Triangles = ti.types.ndarray(dtype=ti.f32, ndim=3)
    Indices = ti.types.ndarray(dtype=ti.i32, ndim=2)
    Order = ti.types.ndarray(dtype=ti.i32, ndim=1)
    Colors = ti.types.ndarray(dtype=ti.u8, ndim=2)


@ti.func
def nearest_even(value: ti.f32) -> ti.i32:
    """
    Match NumPy's rounding at exact half-integer palette and pixel coordinates.
    """
    low = ti.cast(ti.floor(value), ti.i32)
    fraction = value - low
    if fraction > 0.5 or (fraction == 0.5 and low % 2 != 0):
        low += 1
    return low


@ti.kernel
def project(xy: Points, heat: Floats, camera: Floats, vertices: Vertices,
            pixels: Points, depths: Depths):
    """
    Project top and base vertices using the existing orthographic camera and world coordinates.
    """
    for index in range(vertices.shape[0]):
        source = index % heat.shape[0]
        x, y = xy[source, 0], xy[source, 1]
        z = 0.0
        if index < heat.shape[0]:
            z = heat[source] * camera[13]
        vertices[index, 0], vertices[index, 1], vertices[index, 2] = x, y, z
        rx, ry = camera[0] * x + camera[1] * y, camera[2] * x + camera[3] * y
        py = -ry * camera[8] - z * camera[9]
        pixels[index, 0] = (camera[4] * rx + camera[5] * py) * camera[10] + camera[11]
        pixels[index, 1] = (camera[6] * rx + camera[7] * py) * camera[10] + camera[12]
        depths[index] = ry * camera[9] - z * camera[8]


@ti.kernel
def shade(faces: Indices, vertices: Vertices, pixels: Points, depths: Depths,
          heat: Floats, palette: Colors, triangles: Triangles, colors: Colors,
          face_depths: Depths, quantize: ti.i32):
    """
    Compute original flat-face lighting, palette levels and mean painter depths in parallel.
    """
    for face in range(faces.shape[0]):
        a, b, c = faces[face, 0], faces[face, 1], faces[face, 2]
        va = ti.Vector([vertices[a, k] for k in ti.static(range(3))])
        vb = ti.Vector([vertices[b, k] for k in ti.static(range(3))])
        vc = ti.Vector([vertices[c, k] for k in ti.static(range(3))])
        normal = (vb - va).cross(vc - va)
        normal /= ti.max(normal.norm(), 1e-8)
        light = ti.Vector([-0.35, -0.45, 1.0]).normalized()
        brightness = 0.38 + 0.62 * ti.min(1.0, ti.max(0.0, normal.dot(light)))
        level = (heat[a % heat.shape[0]] + heat[b % heat.shape[0]]
                 + heat[c % heat.shape[0]]) / 3.0
        color_index = ti.min(255, ti.max(0, nearest_even(level * 255)))
        for channel in ti.static(range(3)):
            colors[face, channel] = ti.cast(palette[color_index, channel] * brightness, ti.u8)
        for corner, axis in ti.static(ti.ndrange(3, 2)):
            value = pixels[faces[face, corner], axis]
            if quantize:
                value = ti.cast(nearest_even(value), ti.f32)
            triangles[face, corner, axis] = value
        face_depths[face] = (depths[a] + depths[b] + depths[c]) / 3.0


@ti.kernel
def reorder(order: Order, triangles: Triangles, colors: Colors,
            ordered_triangles: Triangles, ordered_colors: Colors):
    """
    Apply the host's stable painter order without reading triangles or colors back to NumPy.
    """
    for rank in range(order.shape[0]):
        face = order[rank]
        for corner, axis in ti.static(ti.ndrange(3, 2)):
            ordered_triangles[rank, corner, axis] = triangles[face, corner, axis]
        for channel in ti.static(range(3)):
            ordered_colors[rank, channel] = colors[face, channel]
