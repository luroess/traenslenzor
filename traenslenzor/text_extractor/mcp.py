import logging

import cv2
import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from numpy.typing import NDArray
from PIL import Image

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import BBoxPoint, ExtractedDocument, SessionState
from traenslenzor.text_extractor.flatten_image import deskew_document
from traenslenzor.text_extractor.ocr import run_ocr

ADDRESS = "127.0.0.1"
PORT = 8002
TEXT_EXTRACTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

text_extractor = FastMCP("Text Extractor")

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

    flattening_result = deskew_document(orig_img)

    flattened_img = orig_img
    transformation_matrix = np.eye(3, dtype=np.float64)
    document_coordinates: NDArray[np.float32] = np.array([])
    if flattening_result is not None:
        logger.info("Image flattening successful")
        flattened_img, transformation_matrix, document_coordinates = flattening_result
    else:
        logger.error("Image flattening failed, proceeding with original image")

    upload_image = cv2.cvtColor(flattened_img, cv2.COLOR_BGR2RGB)
    flattened_image_id = await FileClient.put_img(
        f"{session_id}_deskewed.png", Image.fromarray(upload_image)
    )
    if flattened_image_id is None:
        logger.error("Uploading of extracted document image failed")
        return "Uploading of extracted document image failed"

    extracted_document = ExtractedDocument(
        id=flattened_image_id,
        transformation_matrix=transformation_matrix.tolist(),
        documentCoordinates=[BBoxPoint(x=pt[0], y=pt[1]) for pt in document_coordinates],
    )

    res = run_ocr(flattened_img)

    if res is None:
        raise ToolError("OCR text extraction failed")

    if res is None:
        logger.error("OCR failed to extract text")
        return "OCR failed to extract text"

    def update_session(session: SessionState):
        session.text = res  # pyright: ignore[reportAttributeAccessIssue]
        session.extractedDocument = extracted_document

    await SessionClient.update(session_id, update_session)

    if res is None:
        return "Extracted no text from image"

    return "\n".join([f"{index}: {text.extractedText}" for index, text in enumerate(res)])


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

    deskewed = deskew_document(npimg)
    if deskewed is None:
        print("Error: None")
        exit(-1)
    flattened_img = deskewed[0]
    print(run_ocr(flattened_img))
