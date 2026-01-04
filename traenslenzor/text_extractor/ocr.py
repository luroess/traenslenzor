import logging
from collections import defaultdict
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract

from traenslenzor.file_server.session_state import BBoxPoint, OCRTextItem

logger = logging.getLogger(__name__)


def parse_tesseract_lines(data) -> list[OCRTextItem]:
    """
    Convert pytesseract.image_to_data output into a list of OCRTextItem, grouped by line
    """
    lines = defaultdict(list)
    confs = defaultdict(list)
    bboxes = defaultdict(list)

    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        block, line = data["block_num"][i], data["line_num"][i]
        key = (block, line)

        lines[key].append(text)
        confs[key].append(float(data["conf"][i]))

        left, top, width, height = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        bbox = [
            BBoxPoint(x=left, y=top),
            BBoxPoint(x=left + width, y=top),
            BBoxPoint(x=left + width, y=top + height),
            BBoxPoint(x=left, y=top + height),
        ]
        bboxes[key].append(bbox)

    items = []
    for key in sorted(lines.keys()):
        line_text = " ".join(lines[key])
        avg_conf = sum(confs[key]) / len(confs[key])
        all_x = [pt.x for word_bbox in bboxes[key] for pt in word_bbox]
        all_y = [pt.y for word_bbox in bboxes[key] for pt in word_bbox]
        line_bbox = [
            BBoxPoint(x=min(all_x), y=min(all_y)),
            BBoxPoint(x=max(all_x), y=min(all_y)),
            BBoxPoint(x=max(all_x), y=max(all_y)),
            BBoxPoint(x=min(all_x), y=max(all_y)),
        ]
        items.append(OCRTextItem(extractedText=line_text, confidence=avg_conf, bbox=line_bbox))

    return items


def tesseract_ocr_lines(npimg: np.ndarray) -> list[OCRTextItem]:
    """
    Run Tesseract OCR on a NumPy image (BGR or grayscale) and return results line by line
    """
    # Convert BGR (OpenCV) to RGB for Tesseract
    if npimg.ndim == 3:
        img_rgb = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = npimg

    data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
    return parse_tesseract_lines(data)


def run_ocr(npimg: np.ndarray) -> Optional[Any]:
    try:
        return tesseract_ocr_lines(npimg)
    except Exception as e:
        print(f"Tesseract OCR failed with: {e}, returning None")
        return None

def draw_text_items(npimg: np.ndarray, items: list[OCRTextItem]) -> np.ndarray:
    """
    Draw bounding boxes and extracted text on an image.
    """
    img = npimg.copy()

    for item in items:
        pts = np.array([[pt.x, pt.y] for pt in item.bbox], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        top_left = (int(min(pt.x for pt in item.bbox)), int(min(pt.y for pt in item.bbox)))
        cv2.putText(
            img,
            item.extractedText,
            top_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return img


if __name__ == "__main__":
    import os

    def load_image_as_bytes(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = dir_path + "/../../test_images/test_image_ocr.png"
    npimg = load_image_as_bytes(image_path)
    r = run_ocr(npimg)
    if r:
        print(r)
        rendered = draw_text_items(npimg, r)
        cv2.imshow("OCR Overlay", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("failed to extract any text items")
