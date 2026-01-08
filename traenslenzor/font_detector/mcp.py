"""MCP server for font detection and size estimation."""

import io
import json
import logging
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP
from PIL import Image

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import (
    FontInfo,
    SessionState,
    add_font_info,
)
from traenslenzor.supervisor.config import settings

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
        with Image.open(image_path) as img:
            font_name = detector.detect(img)
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
                with Image.open(image_path) as img:
                    font_name = detector.detect(img)
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
    if not session:
        return "Error: No active session found"

    # Use extracted (flattened) document
    if not session.extractedDocument or not session.extractedDocument.id:
        logger.error("No flattened document found. Text extraction must be run first.")
        return "Error: No flattened document found. Please run text extraction first."

    image_id_to_use = session.extractedDocument.id
    logger.info(f"Using flattened document image: {image_id_to_use}")

    # Download the raw document image
    try:
        # Download image bytes
        image_bytes = await FileClient.get_raw_bytes(image_id_to_use)
        if not image_bytes:
            return "Error: Could not download document image"

        # Load image
        try:
            full_image = Image.open(io.BytesIO(image_bytes))
            if full_image.mode != "RGB":
                full_image = full_image.convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}"

        # Smart Cropping: Find the 224x224 region containing the most text boxes
        crop_image = full_image
        if session.text:
            target_size = 224
            best_window = None
            max_count = -1

            # Convert all bboxes to rectangles (min_x, min_y, max_x, max_y)
            text_rects = []
            for t in session.text:
                if t.bbox and len(t.bbox) == 4:
                    xs = [p.x for p in t.bbox]
                    ys = [p.y for p in t.bbox]
                    text_rects.append((min(xs), min(ys), max(xs), max(ys)))

            if text_rects:
                img_w, img_h = full_image.size

                # If image is smaller than target, use full image
                if img_w <= target_size and img_h <= target_size:
                    best_window = (0, 0, img_w, img_h)
                else:
                    # Search for best window centered on each text box
                    for center_rect in text_rects:
                        cx = (center_rect[0] + center_rect[2]) / 2
                        cy = (center_rect[1] + center_rect[3]) / 2

                        # Define 224x224 window around center
                        w_min_x = max(0, int(cx - target_size / 2))
                        w_min_y = max(0, int(cy - target_size / 2))
                        w_max_x = min(img_w, w_min_x + target_size)
                        w_max_y = min(img_h, w_min_y + target_size)

                        # Shift window if it goes out of bounds (to keep it 224x224 if possible)
                        if w_max_x - w_min_x < target_size and w_min_x > 0:
                            w_min_x = max(0, w_max_x - target_size)
                        if w_max_y - w_min_y < target_size and w_min_y > 0:
                            w_min_y = max(0, w_max_y - target_size)

                        # Count intersections
                        count = 0
                        for r in text_rects:
                            # Check intersection
                            if (
                                r[0] < w_max_x
                                and r[2] > w_min_x
                                and r[1] < w_max_y
                                and r[3] > w_min_y
                            ):
                                count += 1

                        if count > max_count:
                            max_count = count
                            best_window = (w_min_x, w_min_y, w_max_x, w_max_y)

                if best_window:
                    crop_image = full_image.crop(best_window)
                    logger.info(f"Smart crop to dense region: {best_window} with {max_count} boxes")

        # Save debug images
        if settings.llm.debug_mode:
            try:
                debug_dir = Path("debug")
                debug_dir.mkdir(parents=True, exist_ok=True)

                full_image_path = debug_dir / f"font_debug_{session_id}_full.png"
                full_image.save(full_image_path)

                crop_image_path = debug_dir / f"font_debug_{session_id}_crop.png"
                crop_image.save(crop_image_path)

                logger.info(f"Saved debug images to {debug_dir}")
            except Exception as e:
                logger.error(f"Failed to save debug images: {e}")

        # Detect font name globally for the document
        name_detector = get_font_name_detector()
        global_font_name = name_detector.detect(crop_image)

        # Get size estimator
        size_estimator = get_font_size_estimator()

        def update_session(session: SessionState):
            debug_info = []
            if session.text is not None:
                updated_texts = []
                for t in session.text:
                    # Calculate font size for each text item
                    font_size = 12  # Default fallback
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
                            font_size_float = size_estimator.estimate(
                                text_box_size=(width, height),
                                text=t.extractedText,
                                font_name=global_font_name,
                            )
                            font_size = int(font_size_float)
                        except Exception as e:
                            logger.error(f"Error estimating font size: {e}")
                            font_size = 12  # Fallback

                    # Create font info and add to text item
                    font_info = FontInfo(
                        detectedFont=global_font_name,
                        font_size=font_size,
                    )
                    detected_item = add_font_info(t, font_info)
                    updated_texts.append(detected_item)

                    # Collect debug info
                    debug_info.append(
                        {
                            "text": detected_item.extractedText,
                            "bbox": [{"x": p.x, "y": p.y} for p in detected_item.bbox]
                            if detected_item.bbox
                            else None,
                            "detectedFont": detected_item.font.detectedFont,
                            "font_size": detected_item.font.font_size,
                        }
                    )

                session.text = updated_texts

            # Save debug info to file
            if settings.llm.debug_mode:
                try:
                    debug_file = Path("debug") / f"font_debug_{session_id}.json"
                    debug_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_file, "w") as f:
                        json.dump(debug_info, f, indent=2)
                    logger.info(f"Saved font debug info to {debug_file}")
                except Exception as e:
                    logger.error(f"Failed to save font debug info: {e}")

        await SessionClient.update(session_id, update_session)
        return f"Font detection successful. Detected font: {global_font_name}"

    except Exception as e:
        logger.error(f"Font detection failed: {e}")
        return f"Error during font detection: {str(e)}"


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
    await font_detector.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
