"""
Project a relative-heat history grid with bounded CPU arrays and antialiased edges.
"""

import cv2
import numpy as np

from .settings import Settings


class HistoryMesh:
    """
    Cache one view and topology; the horizontal axis is profile position and depth is history.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Fit a fixed 4:3 scene and build a floor without inventing missing history samples.
        """
        self.width = settings.resolution
        self.height = round(settings.resolution * 0.75)
        self.settings = settings
        x, y = np.meshgrid(
            np.linspace(-1, 1, settings.samples, dtype=np.float32),
            np.linspace(0.75, -0.75, settings.history, dtype=np.float32),
        )
        self.xy = np.column_stack((x.ravel(), y.ravel()))
        yaw, tilt, roll = np.deg2rad((settings.yaw, settings.tilt, settings.scene_rotation))
        self.rotation = np.array(
            ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw))), np.float32
        )
        self.roll = np.array(
            ((np.cos(roll), -np.sin(roll)), (np.sin(roll), np.cos(roll))), np.float32
        )
        self.sin_tilt, self.cos_tilt = float(np.sin(tilt)), float(np.cos(tilt))
        self.scale = 1.0
        self.offset = np.zeros(2, np.float32)
        corners = np.array(
            [(x, y, z) for x in (-1, 1) for y in (-0.75, 0.75) for z in (0, settings.height)],
            np.float32,
        )
        projected, _ = self.project(corners)
        low, high = projected.min(axis=0), projected.max(axis=0)
        size = np.array((self.width, self.height), np.float32)
        self.scale = float(np.min(size * 0.82 / np.maximum(high - low, 1e-6))) * settings.zoom
        self.offset = size * 0.5 - (low + high) * self.scale * 0.5
        self.offset += size * np.array((settings.offset_x, settings.offset_y)) / 100
        grid = np.arange(settings.history * settings.samples).reshape(
            settings.history, settings.samples
        )
        a, b = grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel()
        c, d = grid[1:, :-1].ravel(), grid[1:, 1:].ravel()
        self.triangles = np.stack(
            (np.column_stack((a, d, b)), np.column_stack((a, c, d))), axis=1
        ).reshape(-1, 3)
        self.floor = np.full((self.height, self.width, 3), (18, 15, 12), np.uint8)
        for t in np.linspace(-1, 1, 9):
            vertices = np.array(
                ((-1, t * 0.75, 0), (1, t * 0.75, 0), (t, -0.75, 0), (t, 0.75, 0)), np.float32
            )
            points, _ = self.project(vertices)
            lines = np.rint(points).astype(np.int32).reshape(2, 2, 2)
            cv2.polylines(self.floor, list(lines), False, (43, 39, 33), 1, cv2.LINE_AA)

    def project(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply orthographic yaw, elevation, scene rotation, pan and zoom.
        """
        xy = vertices[:, :2] @ self.rotation.T
        z = vertices[:, 2]
        points = np.column_stack((xy[:, 0], -xy[:, 1] * self.sin_tilt - z * self.cos_tilt))
        depth = xy[:, 1] * self.cos_tilt - z * self.sin_tilt
        return (points @ self.roll.T) * self.scale + self.offset, depth

    def render(self, heat: np.ndarray, colors: np.ndarray, wireframe: bool) -> np.ndarray:
        """
        Draw only recorded profiles, with the newest at the front edge of the fixed time grid.
        """
        canvas = self.floor.copy()
        count = len(heat)
        if count == 0:
            return canvas
        samples = self.settings.samples
        start = (self.settings.history - count) * samples
        vertices = np.column_stack((self.xy[start:], heat.ravel() * self.settings.height))
        points, depth = self.project(vertices)
        if count == 1:
            line = np.rint(points).astype(np.int32)
            ink = colors[np.rint(heat[0] * 255).astype(np.uint8)]
            for index in range(samples - 1):
                cv2.line(
                    canvas,
                    tuple(line[index]),
                    tuple(line[index + 1]),
                    tuple(int(v) for v in ink[index]),
                    1,
                    cv2.LINE_AA,
                )
            return canvas
        triangles = self.triangles[self.triangles[:, 0] >= start] - start
        faces = vertices[triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-8)
        light = np.array((-0.35, -0.45, 1), np.float32)
        light /= np.linalg.norm(light)
        shade = 0.38 + 0.62 * np.clip(normals @ light, 0, 1)
        indices = np.rint(heat.ravel()[triangles].mean(axis=1) * 255).astype(np.uint8)
        ink = (colors[indices] * shade[:, None]).astype(np.uint8)
        order = np.argsort(-depth[triangles].mean(axis=1), kind="stable")
        pixels = np.rint(points[triangles][order]).astype(np.int32)
        for triangle, color in zip(pixels, ink[order].tolist(), strict=True):
            cv2.fillConvexPoly(canvas, triangle, color, cv2.LINE_AA)
            if wireframe:
                cv2.polylines(canvas, [triangle], True, (35, 33, 30), 1, cv2.LINE_AA)
        return canvas
