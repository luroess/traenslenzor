import json
import logging

import cv2
import numpy as np
from fastmcp import FastMCP
from PIL import Image

from traenslenzor.file_server.client import FileClient
from traenslenzor.layout_detector.flatten_image import deskew_document
from traenslenzor.layout_detector.paddleocr import run_ocr

ADDRESS = "127.0.0.1"
PORT = 8001
LAYOUT_DETECTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

layout_detector = FastMCP("Layout detector")

logger = logging.getLogger(__name__)


def bytes_to_numpy_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@layout_detector.tool
async def detect_layout(language: str, document_reference: str):
    file_data = await FileClient.get_raw_bytes(document_reference)
    if file_data is None:
        logger.error("Invalid file id, no such document found")
        return f"Document not found: {document_reference}"

    orig_img = bytes_to_numpy_image(file_data)

    flattened_img = deskew_document(orig_img)
    if flattened_img is None:
        logger.error("Image flattening failed, fallback to original.")
        flattened_img = orig_img

    upload_image = cv2.cvtColor(flattened_img, cv2.COLOR_BGR2RGB)
    # TODO: Store in state
    _flattened_image_id = await FileClient.put_img(
        f"{document_reference}_deskewed.png", Image.fromarray(upload_image)
    )

    success, encoded = cv2.imencode(".png", flattened_img)
    if not success:
        logger.error("Failed to encode image for OCR request")
        return "Failed to encode image for OCR"

    res = run_ocr(language, encoded)
    logger.info(res)

    # TODO: store in state
    res = await FileClient.put_bytes("json", json.dumps(res).encode("utf-8"))
    if res is None:
        logger.error("File upload failed for ocr json")
        return "Failed to upload json to server"
    return res


async def run():
    await layout_detector.run_async(transport="streamable-http", port=PORT, host=ADDRESS)
