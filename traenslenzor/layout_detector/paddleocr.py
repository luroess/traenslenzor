import json

from paddleocr import PaddleOCR  # type: ignore
from pydantic import Json


def result_to_json(results) -> Json:
    return json.dumps(
        [
            {"text": text, "confidence": score, "bbox": [list(map(int, pt)) for pt in poly]}
            for r in results
            for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"])
        ]
    )


def paddle_ocr(lang: str, image_path: str) -> Json:
    ocr = PaddleOCR(lang=lang, use_angle_cls=False)
    results = ocr.predict(image_path)
    return results


if __name__ == "__main__":
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = dir_path + "/test_image.png"
    r = paddle_ocr("en", image_path)
    j = result_to_json(r)
    print(j)
