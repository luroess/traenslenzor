import logging

import cv2
import numpy as np
from fastmcp import FastMCP
from PIL import Image

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import ExtractedDocument, SessionState
from traenslenzor.text_extractor.flatten_image import deskew_document
from traenslenzor.text_extractor.paddleocr import run_ocr

ADDRESS = "127.0.0.1"
PORT = 8002
TEXT_EXTRACTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

text_extractor = FastMCP("Layout detector")

logger = logging.getLogger(__name__)


def bytes_to_numpy_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@text_extractor.tool
async def extract_text(session_id: str) -> str:
    """Extracts text from a document.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    """

    logger.info(f"Extracting text in session '{session_id}'")
    session = await SessionClient.get(session_id)

    if session.rawDocumentId is None:
        logger.error(f"No raw document available for session : {session_id}")
        return "No raw document available for this session"

    file_data = await FileClient.get_raw_bytes(session.rawDocumentId)
    if file_data is None:
        logger.error("Invalid file id, no such document found")
        return f"Document not found: {session.rawDocumentId}"

    orig_img = bytes_to_numpy_image(file_data)

    flattened_img = deskew_document(orig_img)
    if flattened_img is None:
        logger.error("Image flattening failed, fallback to original.")
        flattened_img = orig_img

    upload_image = cv2.cvtColor(flattened_img, cv2.COLOR_BGR2RGB)
    flattened_image_id = await FileClient.put_img(
        f"{session_id}_deskewed.png", Image.fromarray(upload_image)
    )
    if flattened_image_id is None:
        logger.error("Uploading of extracted document image failed")
        return "Uploading of extracted document image failed"

    extracted_document = ExtractedDocument(
        id=flattened_image_id,
        # TODO: fix this shit
        documentCoordinates=[],
    )

    res = run_ocr("en", flattened_img)
    logger.info(res)

    def update_session(session: SessionState):
        session.text = res
        session.extractedDocument = extracted_document

    await SessionClient.update(session_id, update_session)
    return "Text extraction successful"


async def run():
    await text_extractor.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )


if __name__ == "__main__":
    import os

    def load_image_as_bytes(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = dir_path + "/../../test_images/skewed_image_1.jpeg"
    npimg = load_image_as_bytes(image_path)
    # run_ocr("en", npimg)

    flattened_img = deskew_document(npimg)
    if flattened_img is None:
        print("Error: None")
        exit(-1)
    print(run_ocr("en", flattened_img))
