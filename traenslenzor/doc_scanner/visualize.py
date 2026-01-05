"""Visualization helpers for deskew outputs."""

from __future__ import annotations

import cv2
import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray as NDArray

from .utils import build_full_image_corners, order_points_clockwise


def draw_polygon_overlay(
    image_rgb: UInt8[NDArray, "H W 3"],
    corners_xy: Float32[NDArray, "4 2"] | None,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> UInt8[NDArray, "H W 3"]:
    """Overlay a polygon on an RGB image.

    Args:
        image_rgb (ndarray[uint8]): Input RGB image, shape (H, W, 3).
        corners_xy (ndarray[float32] | None): Polygon corners in (x, y) order.
        color (tuple[int, int, int]): RGB color for the polygon.
        thickness (int): Line thickness in pixels.

    Returns:
        ndarray[uint8]: Image with polygon overlay, shape (H, W, 3).
    """
    overlay = image_rgb.copy()
    if corners_xy is None:
        corners_xy = build_full_image_corners(overlay.shape[0], overlay.shape[1])
    corners_xy = order_points_clockwise(corners_xy)
    pts = np.round(corners_xy).astype(np.int32).reshape(-1, 1, 2)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.polylines(bgr, [pts], isClosed=True, color=color[::-1], thickness=thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_grid_overlay(
    image_rgb: UInt8[NDArray, "H W 3"],
    map_xy: Float32[NDArray, "H2 W2 2"],
    *,
    step: int = 40,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1,
) -> UInt8[NDArray, "H W 3"]:
    """Overlay a map_xy grid on an RGB image.

    Args:
        image_rgb (ndarray[uint8]): Input RGB image, shape (H, W, 3).
        map_xy (ndarray[float32]): Map of shape (H2, W2, 2) mapping output -> input.
        step (int): Grid step size in output pixels.
        color (tuple[int, int, int]): RGB color for grid lines.
        thickness (int): Line thickness in pixels.

    Returns:
        ndarray[uint8]: Image with grid overlay, shape (H, W, 3).
    """
    overlay = image_rgb.copy()
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    out_h, out_w, _ = map_xy.shape

    h_in, w_in = overlay.shape[:2]
    color_bgr = color[::-1]

    def _clip_points(points: NDArray) -> NDArray:
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, w_in - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, h_in - 1)
        return clipped

    for y in range(0, out_h, step):
        pts = map_xy[y, :, :]
        pts = pts[:: max(1, step // 2)]
        pts = _clip_points(pts)
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [poly], isClosed=False, color=color_bgr, thickness=thickness)

    for x in range(0, out_w, step):
        pts = map_xy[:, x, :]
        pts = pts[:: max(1, step // 2)]
        pts = _clip_points(pts)
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [poly], isClosed=False, color=color_bgr, thickness=thickness)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
