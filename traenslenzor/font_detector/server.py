"""MCP server for font detection and size estimation."""

import json
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

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


@font_detector.tool
def detect_font_name(image_path: str) -> str:
    """Detect font name from an image containing text.

    Args:
        image_path: Path to image file containing text
    """
    try:
        # Detect font name
        detector = get_font_name_detector()
        font_name = detector.detect(image_path)
        return json.dumps({"font_name": font_name})
    except Exception as e:
        return json.dumps({"error": str(e)})


@font_detector.tool
def estimate_font_size(text_box_size: List[float], text: str, font_name: str = "") -> str:
    """Estimate font size in points from text box dimensions and content.

    Args:
        text_box_size: Text box dimensions as [width_px, height_px]
        text: Text content in the box
        font_name: Optional font name hint (if known)
    """
    if not text_box_size or len(text_box_size) != 2:
        return json.dumps({"error": "text_box_size must be a 2-element array"})

    if not text:
        return json.dumps({"error": "text is required"})

    try:
        # If no font name provided, detect it
        if not font_name:
            return json.dumps(
                {
                    "error": "font_name is required (automatic detection from text not yet implemented)"
                }
            )

        # Estimate font size
        estimator = get_font_size_estimator()
        font_size_pt = estimator.estimate(
            text_box_size=tuple(text_box_size),
            text=text,
            font_name=font_name,
        )

        return json.dumps({"font_size_pt": font_size_pt})

    except Exception as e:
        return json.dumps({"error": str(e)})


async def run():
    await font_detector.run_async(transport="streamable-http", port=PORT, host=ADDRESS)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
