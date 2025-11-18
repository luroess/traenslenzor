import json
import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient
from traenslenzor.layout_detector.paddleocr import ocr_from_bytes

ADDRESS = "127.0.0.1"
PORT = 8001
LAYOUT_DETECTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

layout_detector = FastMCP("Layout detector")

logger = logging.getLogger(__name__)


@layout_detector.tool
async def detect_layout(language: str, document_reference: str):
    file_data = await FileClient.get(document_reference)
    if file_data is None:
        logger.error("Invalid file id, no such document found")
        return f"Document not found: {document_reference}"
    res = ocr_from_bytes(language, file_data)
    logger.info(res)

    res = await FileClient.put_bytes("json", json.dumps(res).encode("utf-8"))
    if res is None:
        logger.error("File upload failed for ocr json")
        return "Failed to upload json to server"
    return res


async def run():
    await layout_detector.run_async(transport="streamable-http", port=PORT, host=ADDRESS)
