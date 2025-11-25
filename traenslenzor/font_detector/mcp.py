import logging

from fastmcp import FastMCP

ADDRESS = "127.0.0.1"
PORT = 8003
FONT_DETECTOR_PATH = f"http://{ADDRESS}:{PORT}/mcp"

font_detector = FastMCP("Font Detector")

logger = logging.getLogger(__name__)


@font_detector.tool
async def detect_font(session_id: str) -> str:
    """Detect Fonts from Document.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    """
    return ""


async def run():
    await font_detector.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
