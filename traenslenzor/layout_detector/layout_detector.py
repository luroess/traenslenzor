from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient

ADDRESS = "127.0.0.1"
PORT = 8001
LAYOUT_DETECTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

layout_detector = FastMCP("Layout detector")


@layout_detector.tool
async def detect_layout(document_reference: str):
    file_data = await FileClient.get_raw_bytes(document_reference)
    if file_data is None:
        return f"Document not found: {document_reference}"
    h = hash(file_data)
    print(f"Detected layout for file with hash {h}")


async def run():
    await layout_detector.run_async(transport="streamable-http", port=PORT, host=ADDRESS)
