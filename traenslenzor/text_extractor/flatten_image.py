import logging
import os
<<<<<<< HEAD
from typing import Optional, cast, List
=======
from typing import List, Optional, Tuple, cast
>>>>>>> 20eebb5 ([Text Extractor] formatting)

import cv2
import numpy as np
from numpy.typing import NDArray

from traenslenzor.file_server.session_state import BBoxPoint

logger = logging.getLogger(__name__)


def _order_points(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    pts_f32 = np.asarray(pts, dtype=np.float32)
    y_sorted = pts_f32[np.argsort(pts_f32[:, 1])]
    top_left, top_right = y_sorted[:2][np.argsort(y_sorted[:2][:, 0])]
    bottom_left, bottom_right = y_sorted[2:][np.argsort(y_sorted[2:][:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _warp_to_rectangle(
    image: NDArray[np.uint8], pts: NDArray[np.float32]
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
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
    return (cast(NDArray[np.uint8], warped), np.array(M).astype(np.float64))


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


def deskew_document(
    image: NDArray[np.uint8],
) -> Optional[tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float32]]]:
    pts = find_document_corners(image)
    if pts is None:
        logger.error("No document corners identified in image")
        return None

    flattend_img, matrix = _warp_to_rectangle(image, pts)

    return (flattend_img, matrix, pts)


def mark_corners(image: NDArray[np.uint8], pts: List[BBoxPoint]) -> NDArray[np.uint8]:
    img_marked = image.copy()
    for i, pt in enumerate(pts):
        cv2.circle(img_marked, (int(pt.x), int(pt.y)), 10, (0, 0, 255), -1)
        cv2.putText(
            img_marked,
            str(i + 1),
            (int(pt.x) + 10, int(pt.y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return img_marked

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = dir_path + "/../../test_images/skewed_image_1.jpeg"
    img = cv2.imread(image_path)
    flat, matrix = deskew_document(img)  # type: ignore
    pts, flat = deskew_document(img)  # type: ignore

    if pts is not None:
        img_with_corners = mark_corners(img, pts)  # type: ignore
        cv2.imshow("Corners Marked", img_with_corners)

    if flat is not None:
        cv2.imshow("Deskewed Image", flat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
