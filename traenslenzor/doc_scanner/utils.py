"""Shared geometry helpers for deskew backends."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray as NDArray


def order_points_clockwise(
    pts: Float32[NDArray, "N 2"],
) -> Float32[NDArray, "4 2"]:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (ndarray[float32]): Array of shape (4, 2) with (x, y) coordinates.

    Returns:
        ndarray[float32]: Ordered points with shape (4, 2).
    """
    pts_arr = np.asarray(pts, dtype=np.float32)
    if pts_arr.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts_arr.shape[0]}")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts_arr.sum(axis=1)
    diff = np.diff(pts_arr, axis=1)

    rect[0] = pts_arr[np.argmin(s)]
    rect[2] = pts_arr[np.argmax(s)]
    rect[1] = pts_arr[np.argmin(diff)]
    rect[3] = pts_arr[np.argmax(diff)]
    return rect


def warp_from_corners(
    image_bgr: UInt8[NDArray, "H W 3"],
    corners: Float32[NDArray, "4 2"],
) -> tuple[UInt8[NDArray, "H2 W2 3"], Float32[NDArray, "3 3"], tuple[int, int]]:
    """Warp an image to a rectangle defined by the input corners.

    Args:
        image_bgr (ndarray[uint8]): Input image in BGR order, shape (H, W, 3).
        corners (ndarray[float32]): Corner points in original image coords, shape (4, 2).

    Returns:
        Tuple containing:
            - ndarray[uint8]: Warped image in BGR order, shape (H2, W2, 3).
            - ndarray[float32]: 3x3 homography mapping original -> warped.
            - tuple[int, int]: Output size as (height, width).
    """
    rect = order_points_clockwise(corners)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(round(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(round(max(height_a, height_b)))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, matrix, (max_w, max_h), flags=cv2.INTER_LINEAR)
    return warped, matrix.astype(np.float32), (max_h, max_w)


def build_map_xy_from_homography(
    homography: Float32[NDArray, "3 3"],
    output_size: tuple[int, int],
) -> Float32[NDArray, "H W 2"]:
    """Create a map_xy array from a homography.

    The returned map maps output pixels -> input pixels.

    Args:
        homography (ndarray[float32]): Homography mapping input -> output.
        output_size (tuple[int, int]): Output size as (height, width).

    Returns:
        ndarray[float32]: map_xy of shape (H, W, 2) mapping output -> input.
    """
    out_h, out_w = output_size
    xs, ys = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    grid = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)

    inv = np.linalg.inv(homography)
    mapped = cv2.perspectiveTransform(grid, inv).reshape(out_h, out_w, 2)
    return mapped.astype(np.float32)


def sample_map_xy(
    map_xy: Float32[NDArray, "H W 2"],
    coords_xy: Float32[NDArray, "N 2"],
) -> Float32[NDArray, "N 2"]:
    """Sample map_xy at floating-point coordinates using nearest neighbor.

    Args:
        map_xy (ndarray[float32]): Map of shape (H, W, 2).
        coords_xy (ndarray[float32]): Coordinates in map space, shape (N, 2).

    Returns:
        ndarray[float32]: Mapped coordinates in original space, shape (N, 2).
    """
    pts = np.asarray(coords_xy, dtype=np.float32)
    h, w, _ = map_xy.shape
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return map_xy[yi, xi]


def find_page_corners(
    image_rgb: UInt8[NDArray, "H W 3"],
    *,
    min_area_ratio: float = 0.15,
) -> tuple[Float32[NDArray, "4 2"] | None, float]:
    """Find page corners on an RGB image using thresholding and contours.

    Args:
        image_rgb (ndarray[uint8]): RGB image, shape (H, W, 3).
        min_area_ratio (float): Minimum contour area ratio to accept.

    Returns:
        Tuple[ndarray[float32] | None, float]:
            - Corners array with shape (4, 2) or None if not found.
            - Area ratio of the detected contour to image area.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _thresh, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    best_area = 0.0
    kernel = np.ones((7, 7), np.uint8)

    for candidate in (mask, 255 - mask):
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area > best_area:
            best_area = area
            best = contour

    if best is None:
        return None, 0.0

    hull = cv2.convexHull(best)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect).astype(np.float32).reshape(-1, 1, 2)
    corners = approx.reshape(-1, 2).astype(np.float32)

    img_area = float(image_rgb.shape[0] * image_rgb.shape[1])
    area_ratio = best_area / img_area if img_area > 0 else 0.0
    if area_ratio < min_area_ratio:
        return None, area_ratio

    return order_points_clockwise(corners), area_ratio


def build_full_image_corners(height: int, width: int) -> Float32[NDArray, "4 2"]:
    """Return corners for the full image frame.

    Args:
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        ndarray[float32]: Corners array in clockwise order, shape (4, 2).
    """
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def should_generate_map_xy(output_size: Tuple[int, int], max_pixels: int) -> bool:
    """Decide whether map_xy generation is safe for the given output size."""
    out_h, out_w = output_size
    return out_h * out_w <= max_pixels
