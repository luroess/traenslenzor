import logging

from fastmcp import FastMCP

ADDRESS = "127.0.0.1"
PORT = 8005
TRANSLATOR_PATH = f"http://{ADDRESS}:{PORT}/mcp"

translator = FastMCP("Translator")

logger = logging.getLogger(__name__)


@translator.tool
async def translate(session_id: str) -> str:
    """Translates text.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    """
    return ""


async def run():
    await translator.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
