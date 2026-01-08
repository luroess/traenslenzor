"""Backtransform utilities for deskewed images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray as NDArray

from .utils import order_points_clockwise

if TYPE_CHECKING:
    from traenslenzor.file_server.session_state import BBoxPoint


def _points_to_array(
    points: list["BBoxPoint"] | Float32[NDArray, "4 2"],
) -> Float32[NDArray, "4 2"]:
    if isinstance(points, np.ndarray):
        pts = np.asarray(points, dtype=np.float32)
    else:
        pts = np.array([[point.x, point.y] for point in points], dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 points, got shape {pts.shape}")
    return pts


def backtransform_with_corners(
    extracted_rgb: UInt8[NDArray, "H W 3"],
    document_coordinates: list["BBoxPoint"] | Float32[NDArray, "4 2"],
    output_shape: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> tuple[UInt8[NDArray, "H0 W0 3"], Bool[NDArray, "H0 W0"]]:
    """Project a deskewed image back onto the original canvas using 4 corner points.

    Args:
        extracted_rgb (ndarray[uint8]): Deskewed image, shape (H, W, 3).
        document_coordinates (list[BBoxPoint] | ndarray[float32]): 4 corners in original image coords.
        output_shape (tuple[int, int]): Output (height, width) of the original image.
        interpolation (int): OpenCV interpolation flag for warping.
        border_value (tuple[int, int, int]): Fill color for pixels outside the warp.

    Returns:
        Tuple[ndarray[uint8], ndarray[bool]]:
            - Backtransformed image on the original canvas, shape (H0, W0, 3).
            - Boolean mask of filled pixels, shape (H0, W0).
    """
    h0, w0 = output_shape
    h1, w1 = extracted_rgb.shape[:2]

    dst = order_points_clockwise(_points_to_array(document_coordinates))
    src = np.array(
        [[0.0, 0.0], [float(w1 - 1), 0.0], [float(w1 - 1), float(h1 - 1)], [0.0, float(h1 - 1)]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    back = cv2.warpPerspective(
        extracted_rgb,
        matrix,
        (w0, h0),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    mask_src = np.full((h1, w1), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(  # type: ignore[call-overload]
        mask_src,
        matrix,
        (w0, h0),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return back, mask > 0


def backtransform_with_map_xy(
    extracted_rgb: UInt8[NDArray, "H W 3"],
    map_xy: Float32[NDArray, "H2 W2 2"],
    output_shape: tuple[int, int],
    *,
    upsample: bool = True,
) -> tuple[UInt8[NDArray, "H0 W0 3"], Bool[NDArray, "H0 W0"]]:
    """Project a deskewed image back onto the original canvas using a flow field.

    Args:
        extracted_rgb (ndarray[uint8]): Deskewed image, shape (H, W, 3).
        map_xy (ndarray[float32]): Map of shape (H2, W2, 2) from output -> input pixels.
        output_shape (tuple[int, int]): Output (height, width) of the original image.
        upsample (bool): Whether to upsample map_xy to extracted resolution when needed.

    Returns:
        Tuple[ndarray[uint8], ndarray[bool]]:
            - Backtransformed image on the original canvas, shape (H0, W0, 3).
            - Boolean mask of filled pixels, shape (H0, W0).
    """
    h0, w0 = output_shape
    h1, w1 = extracted_rgb.shape[:2]

    if upsample and map_xy.shape[:2] != (h1, w1):
        map_x = cv2.resize(map_xy[..., 0], (w1, h1), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_xy[..., 1], (w1, h1), interpolation=cv2.INTER_LINEAR)
        map_xy_full = np.stack([map_x, map_y], axis=-1).astype(np.float32)
    else:
        map_xy_full = map_xy.astype(np.float32)

    x_t = np.rint(map_xy_full[..., 0]).astype(np.int32)
    y_t = np.rint(map_xy_full[..., 1]).astype(np.int32)

    valid = (0 <= x_t) & (x_t < w0) & (0 <= y_t) & (y_t < h0)

    back = np.zeros((h0, w0, 3), dtype=np.uint8)
    back[y_t[valid], x_t[valid]] = extracted_rgb[valid]

    mask = np.zeros((h0, w0), dtype=bool)
    mask[y_t[valid], x_t[valid]] = True
    return back, mask
