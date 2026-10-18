"""
Sparse annotation strokes and fast solid fills with measurement-compatible contrast and alpha.
"""

import cv2
import numpy as np


def blend(image: np.ndarray, mask: np.ndarray, color: str, opacity: float) -> None:
    """
    Composite only covered pixels; use OpenCV for dense fills and retain exact antialiased edges.
    """
    if opacity == 0:
        return
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        return
    region, local = image[y:y + height, x:x + width], mask[y:y + height, x:x + width]
    covered = local > 0
    rgb = None if color == "dynamic" else np.array([int(color[i:i + 2], 16) for i in (5, 3, 1)], np.uint8)
    if rgb is not None and cv2.countNonZero(local) > local.size // 3:
        edge = covered & (local != 255)
        original_edges = region[edge]
        target = np.empty_like(region)
        target[:] = rgb
        blended = cv2.addWeighted(region, 1 - opacity, target, opacity, 0)
        cv2.copyTo(blended, local, region)
        if np.any(edge):
            alpha = (local[edge].astype(np.float32) * (opacity / 255))[:, None]
            region[edge] = np.rint(original_edges * (1 - alpha) + rgb * alpha).astype(np.uint8)
        return
    pixels = region[covered]
    target = 255 - pixels if rgb is None else rgb
    alpha = (local[covered].astype(np.float32) * (opacity / 255))[:, None]
    region[covered] = np.rint(pixels * (1 - alpha) + target * alpha).astype(np.uint8)


def contrast(image: np.ndarray, mask: np.ndarray, opacity: float, *, text: bool = False) -> None:
    """
    Match measurement black/white text and inverted haloed borders without full-box float arrays.
    """
    if opacity == 0:
        return
    kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) if text
              else np.ones((3, 3), np.uint8))
    halo = np.asarray(cv2.dilate(mask, kernel))
    covered = halo > 0
    if not np.any(covered):
        return
    pixels = image[covered]
    halo_alpha = (halo[covered].astype(np.float32) / 255)[:, None]
    alpha = (mask[covered].astype(np.float32) / 255)[:, None]
    if text:
        target = pixels * (1 - halo_alpha)
        target = target * (1 - alpha) + 255 * alpha
    else:
        inverted = 255 - pixels
        luminance = np.asarray(cv2.cvtColor(inverted[:, None, :], cv2.COLOR_BGR2GRAY)).ravel()
        outline = np.where(luminance >= 128, 0, 255).astype(np.uint8)[:, None]
        target = pixels * (1 - halo_alpha) + outline * halo_alpha
        target = target * (1 - alpha) + inverted * alpha
    target = np.rint(target).astype(np.uint8)
    blended = cv2.addWeighted(pixels[:, None, :], 1 - opacity, target[:, None, :], opacity, 0)
    image[covered] = np.asarray(blended)[:, 0, :]
