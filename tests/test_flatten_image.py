import cv2
import numpy as np

from traenslenzor.text_extractor.flatten_image import (
    _find_document_corners_threshold,
    find_document_corners,
)


def _make_rect_image() -> np.ndarray:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 40), (270, 180), (255, 255, 255), thickness=-1)
    return img


def test_find_document_corners_threshold_detects_rectangle() -> None:
    img = _make_rect_image()
    corners = _find_document_corners_threshold(img, min_area_ratio=0.1)
    assert corners is not None
    assert corners.shape == (4, 2)


def test_find_document_corners_detects_rectangle() -> None:
    img = _make_rect_image()
    corners = find_document_corners(img)
    assert corners is not None
    assert corners.shape == (4, 2)
