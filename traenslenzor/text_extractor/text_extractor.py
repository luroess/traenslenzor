import json
import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient
from traenslenzor.text_extractor.paddleocr import ocr_from_bytes

ADDRESS = "127.0.0.1"
PORT = 8001
TEXT_EXTRACTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

text_edtractor = FastMCP("Layout detector")

logger = logging.getLogger(__name__)


@text_edtractor.tool
async def extract_text(document_reference: str) -> str:
    """Extracts text from a document.
    Args:
        language (str, optional): Language of the original document.
        document_reference (str): ID of the document to process (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    Returns:
        str: ID of the new document containing the extracted text.
    """
    logger.info(f"Extracting text from '{document_reference}'")
    file_data = await FileClient.get(document_reference)
    if file_data is None:
        logger.error("Invalid file id, no such document found")
        return f"Document not found: {document_reference}"
    res = ocr_from_bytes("en", file_data)
    logger.info(res)

    res = await FileClient.put_bytes("json", json.dumps(res).encode("utf-8"))
    if res is None:
        logger.error("File upload failed for ocr json")
        return "Failed to upload json to server"
    return res


async def run():
    await text_edtractor.run_async(transport="streamable-http", port=PORT, host=ADDRESS)
