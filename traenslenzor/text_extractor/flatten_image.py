import logging
import os
from typing import Optional, cast

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _find_document_corners_threshold(
    image: NDArray[np.uint8],
    *,
    min_area_ratio: float = 0.2,
) -> Optional[NDArray[np.float32]]:
    """Find document corners using a threshold-based contour heuristic.

    This is useful for clean, high-contrast inputs (e.g. synthetic rectangles) and
    serves as a lightweight alternative to edge-based detection.

    Args:
        image: Input image in BGR with shape (H, W, 3).
        min_area_ratio: Minimum contour area ratio (relative to image area) required
            to accept a detected quadrilateral.

    Returns:
        Array of 4 corner points with shape (4, 2) ordered as
        (top-left, top-right, bottom-right, bottom-left), or None if no suitable
        quadrilateral is found.
    """

    if not 0.0 < min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be in (0, 1].")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayblurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _thresh_val, thresh = cv2.threshold(
        grayblurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_refined = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(thresh_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    img_area = w * h

    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < min_area_ratio * img_area:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        return _order_points(pts)

    return None


def _order_points(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    pts_f32 = np.asarray(pts, dtype=np.float32)
    y_sorted = pts_f32[np.argsort(pts_f32[:, 1])]
    top_left, top_right = y_sorted[:2][np.argsort(y_sorted[:2][:, 0])]
    bottom_left, bottom_right = y_sorted[2:][np.argsort(y_sorted[2:][:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _warp_to_rectangle(image: NDArray[np.uint8], pts: NDArray[np.float32]) -> NDArray[np.uint8]:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(round(max(heightA, heightB)))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return cast(NDArray[np.uint8], warped)


def find_document_corners(image: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make grayscale
    grayblurred = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise

    edges = cv2.Canny(grayblurred, 50, 150)  # get edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_refined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges_refined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    img_area = w * h

    for c in cnts[:10]:  #  10 largest contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # contour simplification
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 0.2 * img_area:  # only take contours 20% or larger of total size
                pts = approx.reshape(4, 2).astype(np.float32)
                return _order_points(pts)
    return None


def deskew_document(image: NDArray[np.uint8]) -> Optional[NDArray[np.uint8]]:
    pts = find_document_corners(image)
    if pts is None:
        logger.error("No document corners identified in image")
        return None
    return _warp_to_rectangle(image, pts)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = dir_path + "/../../test_images/skewed_image_1.jpeg"
    img = cv2.imread(image_path)
    flat = deskew_document(img)  # type: ignore
    cv2.imshow("Image", flat)  # type: ignore
    cv2.waitKey(0)
    cv2.destroyAllWindows()
