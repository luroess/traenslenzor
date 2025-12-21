"""MCP server for font detection and size estimation."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import SessionState

from .font_name_detector import FontNameDetector
from .font_size_model.infer import FontSizeEstimator

ADDRESS = "127.0.0.1"
PORT = 8003
FONT_DETECTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

logger = logging.getLogger(__name__)

# Initialize server
font_detector = FastMCP("Font Detector")

# Global instances (lazy loaded)
font_name_detector_instance: Optional[FontNameDetector] = None
font_size_estimator_instance: Optional[FontSizeEstimator] = None


def get_font_name_detector() -> FontNameDetector:
    """Get or create font name detector instance."""
    global font_name_detector_instance
    if font_name_detector_instance is None:
        font_name_detector_instance = FontNameDetector()
    return font_name_detector_instance


def get_font_size_estimator() -> FontSizeEstimator:
    """Get or create font size estimator instance."""
    global font_size_estimator_instance
    if font_size_estimator_instance is None:
        # Use checkpoints in the font_detector directory
        checkpoints_dir = Path(__file__).parent / "checkpoints"
        font_size_estimator_instance = FontSizeEstimator(str(checkpoints_dir))
    return font_size_estimator_instance


def detect_font_name_logic(image_path: str) -> str:
    """Core logic for font name detection."""
    try:
        # Detect font name
        detector = get_font_name_detector()
        font_name = detector.detect(image_path)
        return json.dumps({"font_name": font_name})
    except Exception as e:
        return json.dumps({"error": str(e)})


@font_detector.tool
def detect_font_name(image_path: str) -> str:
    """Detect font name from an image containing text.

    Args:
        image_path: Path to image file containing text
    """
    return detect_font_name_logic(image_path)


def estimate_font_size_logic(
    text_box_size: List[float],
    text: str,
    image_path: str = "",
    font_name: str = "",
    # num_lines: int | None = None, <-- DISABLED
) -> str:
    """Core logic for font size estimation."""
    if not text_box_size or len(text_box_size) != 2:
        return json.dumps({"error": "text_box_size must be a 2-element array"})

    if not text:
        return json.dumps({"error": "text is required"})

    # Calculate lines if not provided
    # if num_lines is None:
    #     num_lines = text.count("\n") + 1 if text else 1

    try:
        # If no font name provided, detect it
        if not font_name:
            if image_path:
                detector = get_font_name_detector()
                font_name = detector.detect(image_path)
            else:
                return json.dumps(
                    {
                        "error": "font_name is required or image_path must be provided for automatic detection"
                    }
                )

        # Estimate font size
        estimator = get_font_size_estimator()
        font_size_pt = estimator.estimate(
            text_box_size=tuple(text_box_size),
            text=text,
            font_name=font_name,
            # num_lines=num_lines,
        )

        return json.dumps(
            {"font_size_pt": font_size_pt, "font_name": font_name}  # Removed num_lines
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@font_detector.tool
def estimate_font_size(
    text_box_size: List[float],
    text: str,
    image_path: str = "",
    font_name: str = "",
    # num_lines: int | None = None, <-- DISABLED
) -> str:
    """Estimate font size in points from text box dimensions and content.

    Args:
        text_box_size: Text box dimensions as [width_px, height_px]
        text: Text content in the box
        image_path: Optional path to image file (used to detect font if font_name not provided)
        font_name: Optional font name hint (if known)
        # num_lines: Number of lines in the text (optional, calculated from text if not provided) <-- DISABLED
    """
    return estimate_font_size_logic(text_box_size, text, image_path, font_name)


async def detect_font_logic(session_id: str) -> str:
    """Core logic for font detection."""
    # Get session state to access raw document ID
    session = await SessionClient.get(session_id)
    if not session or not session.rawDocumentId:
        return "Error: No active session or raw document found"

    # Download the raw document image
    image_path = None
    try:
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = tmp.name

        # Download image bytes and save to temp file
        image_bytes = await FileClient.get_raw_bytes(session.rawDocumentId)
        if not image_bytes:
            return "Error: Could not download raw document image"

        with open(image_path, "wb") as f:
            f.write(image_bytes)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Detect font name globally for the document
        name_detector = get_font_name_detector()
        global_font_name = name_detector.detect(image_path)

        # Get size estimator
        size_estimator = get_font_size_estimator()

        def update_session(session: SessionState):
            if session.text is not None:
                for t in session.text:
                    # Use global font name for all text items
                    t.detectedFont = global_font_name

                    # Estimate font size for each text item
                    if t.bbox and len(t.bbox) == 4:
                        # Calculate width and height from bbox points
                        # bbox is [UL, UR, LR, LL]
                        # Width: distance between UL and UR
                        width = (
                            (t.bbox[1].x - t.bbox[0].x) ** 2 + (t.bbox[1].y - t.bbox[0].y) ** 2
                        ) ** 0.5
                        # Height: distance between UL and LL
                        height = (
                            (t.bbox[3].x - t.bbox[0].x) ** 2 + (t.bbox[3].y - t.bbox[0].y) ** 2
                        ) ** 0.5

                        # Estimate size
                        try:
                            # Default to 1 line if not specified (could be improved by analyzing text)
                            # num_lines = t.extractedText.count("\n") + 1 if t.extractedText else 1

                            font_size = size_estimator.estimate(
                                text_box_size=(width, height),
                                text=t.extractedText,
                                font_name=global_font_name,
                                # num_lines=num_lines,
                            )
                            t.font_size = int(font_size)
                        except Exception as e:
                            logger.error(f"Error estimating font size: {e}")
                            t.font_size = 12  # Fallback
                    else:
                        t.font_size = 12  # Fallback

        await SessionClient.update(session_id, update_session)
        return f"Font detection successful. Detected font: {global_font_name}"

    except Exception as e:
        logger.error(f"Font detection failed: {e}")
        return f"Error during font detection: {str(e)}"
    finally:
        # Cleanup temporary file
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)


@font_detector.tool
async def detect_font(session_id: str) -> str:
    """
    Detect fonts and estimate font sizes for text in a document session.

    This tool downloads the raw document image associated with the session,
    detects the global font used in the document, and then estimates the
    font size for each text element found in the session state.
    The session state is updated with the detected font name and size.

    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").

    Returns:
        str: A message indicating success or failure, including the detected font name.
    """
    return await detect_font_logic(session_id)


async def run():
    await font_detector.run_async(transport="streamable-http", port=PORT, host=ADDRESS)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
