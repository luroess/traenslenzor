"""MCP server for font detection and size estimation."""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient

from .font_name_detector import FontNameDetector
from .font_size_model.infer import FontSizeEstimator

ADDRESS = "127.0.0.1"
PORT = 8003
FONT_DETECTOR_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

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
    num_lines: int | None = None,
) -> str:
    """Core logic for font size estimation."""
    if not text_box_size or len(text_box_size) != 2:
        return json.dumps({"error": "text_box_size must be a 2-element array"})

    if not text:
        return json.dumps({"error": "text is required"})

    # Calculate lines if not provided
    if num_lines is None:
        num_lines = text.count("\n") + 1 if text else 1

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
            num_lines=num_lines,
        )

        return json.dumps(
            {"font_size_pt": font_size_pt, "font_name": font_name, "num_lines": num_lines}
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@font_detector.tool
def estimate_font_size(
    text_box_size: List[float],
    text: str,
    image_path: str = "",
    font_name: str = "",
    num_lines: int | None = None,
) -> str:
    """Estimate font size in points from text box dimensions and content.

    Args:
        text_box_size: Text box dimensions as [width_px, height_px]
        text: Text content in the box
        image_path: Optional path to image file (used to detect font if font_name not provided)
        font_name: Optional font name hint (if known)
        num_lines: Number of lines in the text (optional, calculated from text if not provided)
    """
    return estimate_font_size_logic(text_box_size, text, image_path, font_name, num_lines)


@font_detector.tool()
async def detect_font(
    image_id: str,
    text: str,
    bbox: list[int] = None,
    lines: int | None = None,
) -> str:
    """
    Detects the font family and estimates font size for text in an image.

    Args:
        image_id: The ID of the image file in the file server.
        text: The text content within the bounding box.
        bbox: The bounding box [x, y, w, h] (optional).
        lines: Number of lines of text. If not provided, calculated from text.
    """
    # Logic to handle optional lines parameter
    if lines is None:
        # Calculate lines based on newlines in the text, defaulting to 1
        lines = text.count("\n") + 1 if text else 1

    image_path = None
    try:
        # Download image from file server
        async with FileClient() as client:
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_path = tmp.name

            await client.download_file(image_id, image_path)

        # Detect font name
        name_detector = get_font_name_detector()
        font_name = name_detector.detect(image_path)

        # Estimate font size
        font_size = 0.0
        if bbox and len(bbox) == 4:
            width, height = bbox[2], bbox[3]
            estimator = get_font_size_estimator()
            font_size = estimator.estimate(
                text_box_size=(width, height),
                text=text,
                font_name=font_name,
                num_lines=lines,
            )

        return json.dumps({"font_name": font_name, "font_size": font_size, "lines": lines})

    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        # Cleanup temporary file
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)


async def run():
    await font_detector.run_async(transport="streamable-http", port=PORT, host=ADDRESS)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
