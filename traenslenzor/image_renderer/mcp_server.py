"""FastMCP server for image rendering and text replacement."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import RenderReadyItem, SessionState
from traenslenzor.image_renderer.image_rendering import ImageRenderer
from traenslenzor.supervisor.config import settings

ADDRESS = "127.0.0.1"
PORT = 8006
IMAGE_RENDERER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

# Debug configuration
DEBUG_DIR = "./debug"

# Initialize FastMCP server
image_renderer_mcp = FastMCP("Image Renderer")

# Singleton instance for model persistence
_renderer_instance: ImageRenderer | None = None

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """Result of a text replacement operation."""

    success: bool
    rendered_document_id: str


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
async def replace_text(
    session_id: Annotated[
        str,
        Field(
            description=(
                "The unique identifier for the session containing extracted and translated text. "
                "Must be a valid UUID format (e.g., 'c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4'). "
                "The session must have text extracted and translated before rendering."
            )
        ),
    ],
) -> RenderResult:
    """
    Replace extracted text in an image with translated text using AI-powered inpainting.

    This tool takes a session with extracted and translated text items, loads the original
    image, and uses an AI model to intelligently replace the original text with translations
    while preserving the image's visual style and layout.

    Prerequisites:
        - Session must exist with the given ID
        - Image must have extracted text (via text extraction tool)
        - All text items must be translated (via translation tool)
        - Original document must be available in the session

    Args:
        session_id: The unique identifier for the session containing extracted and translated text.

    Returns:
        RenderResult: Object containing:
            - success: Boolean indicating if the rendering was successful
            - rendered_document_id: The ID of the newly created rendered document

    Raises:
        ToolError: If session is invalid, text is not extracted/translated, or document is missing.
    """
    logger.info(f"Extracting text in session '{session_id}'")
    session = await SessionClient.get(session_id)

    if session.text is None or len(session.text) == 0:
        logger.error("No text items found in session")
        raise ToolError(
            "No text items found in session. Please extract text from the document first."
        )

    # Validate all text items are render ready (have both font and translation)
    if not all(text.type == "render_ready" for text in session.text):
        not_ready = [text.extractedText for text in session.text if text.type != "render_ready"]
        logger.error(f"Text items are not render ready yet: {not_ready}")
        raise ToolError(
            f"All text items must be render ready (with font detection and translation) before rendering. "
            f"Found {len(not_ready)} item(s) not ready. "
            f"Please ensure font detection and translation are complete."
        )

    # Type narrowing: we know all items are render ready now
    render_ready_texts: list[RenderReadyItem] = [
        text for text in session.text if text.type == "render_ready"
    ]
    if session.rawDocumentId is None:
        logger.error(f"No raw document available for session: {session_id}")
        raise ToolError(
            "No raw document available for this session. Please upload a document first."
        )

    if session.extractedDocument is None:
        logger.error(f"No extracted document available for session: {session_id}")
        raise ToolError(
            "No extracted document available. Please extract text from the document first."
        )

    # Load image from FileClient
    image = await FileClient.get_image(session.extractedDocument.id)
    if image is None:
        logger.error(f"Failed to load image for document: {session.extractedDocument}")
        raise ToolError(
            "Failed to load the extracted document image. "
            "The document may have been deleted or is corrupted."
        )

    # Process image
    renderer = get_renderer()
    result_image = await renderer.replace_text(
        image=image,
        texts=render_ready_texts,
        save_debug=settings.llm.debug_mode,
        debug_dir=DEBUG_DIR,
    )

    transformation_matrix = session.extractedDocument.transformation_matrix
    original_image = await FileClient.get_image(session.rawDocumentId)
    if original_image is None:
        logger.error(f"Failed to load original image for document: {session.rawDocumentId}")
        raise ToolError(
            "Failed to load the original document image. "
            "The document may have been deleted or is corrupted."
        )
    original_size = (original_image.width, original_image.height)

    final_image = renderer.transform_image(
        result_image, np.linalg.inv(np.array(transformation_matrix)), original_size
    )

    final_image = renderer.paste_replaced_to_original(original_image, final_image)

    if settings.llm.debug_mode:
        debug_path = Path(DEBUG_DIR)
        debug_path.mkdir(parents=True, exist_ok=True)
        final_image.save(debug_path / "debug-result.png")

    final_id = await FileClient.put_img(f"{session_id}_rendered_img", final_image)
    if final_id is None:
        logger.error("Failed to save rendered image to file server")
        raise ToolError("Failed to save the rendered image. Please try again.")

    def update_session(session: SessionState):
        session.renderedDocumentId = final_id

    await SessionClient.update(session_id, update_session)

    return RenderResult(success=True, rendered_document_id=final_id)


async def run():
    """Run the FastMCP server."""
    await image_renderer_mcp.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
