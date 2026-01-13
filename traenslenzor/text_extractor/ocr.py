import logging
from collections import defaultdict
from statistics import median
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract

from traenslenzor.file_server.session_state import BBoxPoint, OCRTextItem
from traenslenzor.text_extractor.flatten_image import deskew_document

logger = logging.getLogger(__name__)


def parse_tesseract_lines(data) -> list[OCRTextItem]:
    """
    Convert pytesseract.image_to_data output into a list of OCRTextItem, grouped by line
    """
    # Notes:
    # - Tesseract's (block_num, line_num) is not always unique enough; line_num may reset per paragraph.
    # - Even within a "line", Tesseract can sometimes span separate columns/snippets on the same y.
    #   We mitigate this by splitting a line into segments when there is a large horizontal gap.
    words_by_line = defaultdict(list)

    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        page = data.get("page_num", [0] * n)[i]
        block = data.get("block_num", [0] * n)[i]
        par = data.get("par_num", [0] * n)[i]
        line = data.get("line_num", [0] * n)[i]
        key = (page, block, par, line)

        conf_raw = data.get("conf", ["-1"] * n)[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        left, top, width, height = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        right = left + width
        bottom = top + height
        bbox = [
            BBoxPoint(x=left, y=top),
            BBoxPoint(x=right, y=top),
            BBoxPoint(x=right, y=bottom),
            BBoxPoint(x=left, y=bottom),
        ]

        words_by_line[key].append(
            {
                "text": text,
                "conf": conf,
                "bbox": bbox,
                "left": float(left),
                "top": float(top),
                "right": float(right),
                "bottom": float(bottom),
                "width": float(width),
                "height": float(height),
            }
        )

    items = []
    for key in sorted(words_by_line.keys()):
        words = sorted(words_by_line[key], key=lambda w: (w["top"], w["left"]))
        if not words:
            continue

        widths = [w["width"] for w in words if w["width"] > 0]
        heights = [w["height"] for w in words if w["height"] > 0]
        med_width = median(widths) if widths else 0.0
        med_height = median(heights) if heights else 0.0

        # If two consecutive words are far apart horizontally, they're probably different snippets/columns.
        # Threshold is tuned to be conservative: split only on clearly large gaps.
        gap_threshold = max(40.0, 2.5 * med_height, 1.5 * med_width)

        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        prev_right: float | None = None
        prev_center_y: float | None = None
        y_tol = max(6.0, 0.8 * med_height) if med_height > 0 else 12.0

        for w in words:
            center_y = 0.5 * (w["top"] + w["bottom"])
            if current and prev_right is not None and prev_center_y is not None:
                gap = w["left"] - prev_right
                y_delta = abs(center_y - prev_center_y)
                if gap > gap_threshold or y_delta > y_tol:
                    segments.append(current)
                    current = []
                    prev_right = None
                    prev_center_y = None

            current.append(w)
            prev_right = max(prev_right or w["right"], w["right"])
            prev_center_y = center_y

        if current:
            segments.append(current)

        for seg in segments:
            seg_text = " ".join(w["text"] for w in seg)
            seg_confs = [w["conf"] for w in seg if w["conf"] >= 0]
            avg_conf = (sum(seg_confs) / len(seg_confs)) if seg_confs else 0.0

            all_x = [pt.x for w in seg for pt in w["bbox"]]
            all_y = [pt.y for w in seg for pt in w["bbox"]]
            line_bbox = [
                BBoxPoint(x=min(all_x), y=min(all_y)),
                BBoxPoint(x=max(all_x), y=min(all_y)),
                BBoxPoint(x=max(all_x), y=max(all_y)),
                BBoxPoint(x=min(all_x), y=max(all_y)),
            ]
            items.append(OCRTextItem(extractedText=seg_text, confidence=avg_conf, bbox=line_bbox))

    # Make ordering stable for downstream tooling/visualization
    def _sort_key(item: OCRTextItem) -> tuple[float, float]:
        xs = [pt.x for pt in item.bbox]
        ys = [pt.y for pt in item.bbox]
        return (min(ys) if ys else 0.0, min(xs) if xs else 0.0)

    return sorted(items, key=_sort_key)


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
    image_path = dir_path + "/../../letter.png"
    npimg = load_image_as_bytes(image_path)
    deskew_res = deskew_document(npimg)
    if deskew_res is None:
        print("failed to deskew document")
        exit(1)
    deskewed = deskew_res[0]
    r = run_ocr(deskewed)
    if r:
        print(r)
        rendered = draw_text_items(deskewed, r)
        cv2.imshow("OCR Overlay", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("failed to extract any text items")
