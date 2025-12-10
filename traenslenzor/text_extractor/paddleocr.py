import ast  # noqa: I001 // Do not sort imports or this wont work
import inspect
import logging
import re
from typing import Any, Optional, Sequence

import cv2
import numpy as np

# This is an ugly fix to make paddleocr compatible with langchain 1.0.0
# https://github.com/PaddlePaddle/PaddleOCR/issues/16711#issuecomment-3446427004
# This must be imported prior to paddleocr
from traenslenzor.file_server.session_state import BBoxPoint, OCRTextItem
from traenslenzor.logger import setup_logger
import traenslenzor.text_extractor.shim_langchain_backcomp  # noqa: F401
from paddleocr import PaddleOCR
from pydantic import Json


logger = logging.getLogger(__name__)


def parse_result(results) -> Any:
    return [
        OCRTextItem(
            extractedText=text,
            confidence=score,
            bbox=[BBoxPoint(x=int(pt[0]), y=int(pt[1])) for pt in poly],
        )
        for r in results
        for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"])
    ]


def paddle_ocr(lang: str, image: np.ndarray) -> Json:
    ocr = PaddleOCR(lang=lang, use_angle_cls=False)
    results = ocr.predict(image)

    # paddle messes with log setup...
    # https://github.com/PaddlePaddle/Paddle/pull/76699
    setup_logger()
    return results


def get_paddle_supported_langs() -> list[str]:
    # this is as ugly as it gets but paddle does not offer any api and the values are just hardcoded
    # we could try by exception but this sucks imho
    src = inspect.getsource(PaddleOCR._get_ocr_model_names)

    found = re.findall(r"([A-Z_]+(?:_LANGS|_LANGS)|SPECIFIC_LANGS)\s*=\s*(\[[^\]]*\])", src)
    langs = set()
    for _name, list_literal in found:
        langs.update(ast.literal_eval(list_literal))

    # hardcoded manually checked...
    langs.update(["ch", "en", "japan", "korean", "th", "el", "te", "ta"])

    # sort from short to long.
    sorted_langs = sorted(langs, key=len)
    return sorted_langs


def normalize_language(lang: str) -> str:
    langs = get_paddle_supported_langs()
    lang = lang.lower().strip()

    def find_substr(substr: str, sset: Sequence[str]) -> Optional[str]:
        return next((s for s in sset if s.startswith(substr)), None)

    if flang := find_substr(lang, langs):
        logger.info(f'Found language: "{flang}" for selected language: "{lang}": using "{flang}"')
        return flang

    # lets try the first two chars
    if flang := find_substr(lang[:2], langs):
        logger.info(f'Found language: "{flang}" for selected language: "{lang}": using "{flang}"')
        return flang

    logger.warning(f'no language found for {lang}. Using "latin" as default')
    return "latin"


def run_ocr(lang: str, image: np.ndarray) -> Optional[Any]:
    lang = normalize_language(lang)
    try:
        ocr_res = paddle_ocr(lang, image)
        return parse_result(ocr_res)
    except Exception as e:
        print(f"PaddleOCR failed with: {e}, returning None")
        return None


if __name__ == "__main__":
    import os

    def load_image_as_bytes(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = dir_path + "/../../test_images/test_image_ocr.png"
    npimg = load_image_as_bytes(image_path)
    lang = normalize_language("hi")
    r = paddle_ocr(lang, npimg)
    j = parse_result(r)
    print(j)
