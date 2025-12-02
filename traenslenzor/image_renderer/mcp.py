"""FastMCP server for image rendering and text replacement."""

import logging
from pathlib import Path

import torch
from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.image_renderer.image_renderer import ImageRenderer

ADDRESS = "127.0.0.1"
PORT = 8006
IMAGE_RENDERER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

# Initialize FastMCP server
image_renderer_mcp = FastMCP("Image Renderer")

# Singleton instance for model persistence
_renderer_instance: ImageRenderer | None = None

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return str(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        return str(torch.device("mps"))
    else:
        return str(torch.device("cpu"))


def get_renderer() -> ImageRenderer:
    """Get or create the singleton ImageRenderer instance."""
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = ImageRenderer(device=get_device())
    return _renderer_instance


@image_renderer_mcp.tool
async def replace_text(session_id: str) -> str:
    """
    Replace text in an image using inpainting.

    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    """

    # Debug Config
    debug_dir = "./debug"
    save_debug = True

    logger.info(f"Extracting text in session '{session_id}'")
    session = await SessionClient.get(session_id)

    if session.rawDocumentId is None:
        logger.error(f"No raw document available for session : {session_id}")
        return "No raw document available for this session"

    # Load image from FileClient
    image = await FileClient.get_image(session.rawDocumentId)
    if image is None:
        logger.error("Invalid file id, no such document found")
        return f"Document not found: {session.rawDocumentId}"

    texts = []  # TODO: @BENE extract from session. Please take a look at the paddleocr boxes we get returned :)

    # Process image
    renderer = get_renderer()
    result_image = await renderer.replace_text(
        image=image,
        texts=texts,
        inverse_transformation=None,
        save_debug=save_debug,
        debug_dir=debug_dir,
    )

    if save_debug:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        result_image.save(debug_path / "debug-result.png")

    result_id = await FileClient.put_img(f"{session_id}_rendered_img", result_image)
    assert result_id is not None, "Failed to save result image"

    return result_id


async def run():
    """Run the FastMCP server."""
    await image_renderer_mcp.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
