"""Text rendering and mask creation utilities for image processing."""

import logging
from typing import List

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.Image import Resampling
from PIL.ImageFont import FreeTypeFont
from PIL.ImageFont import ImageFont as ImageFontType

from traenslenzor.file_server.session_state import BBoxPoint, TranslatedTextItem
from traenslenzor.image_utils.image_utils import np_img_to_pil, pil_to_numpy

logger = logging.getLogger(__name__)


def get_angle_from_bbox(bbox: List[BBoxPoint]) -> tuple[float, NDArray[np.float64]]:
    """
    calculates the angle of rotation from a bounding box
    """
    ul = bbox[0]
    ur = bbox[1]

    delta_x = ur.x - ul.x
    delta_y = ur.y - ul.y

    radians = np.atan2(delta_y, delta_x)
    transformation_matrix = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )

    return (np.degrees(radians) % 360, transformation_matrix)


def create_mask(texts: list[TranslatedTextItem], mask_shape: tuple[int, int]) -> NDArray[np.uint8]:
    """
    Create a binary mask from text regions.

    Args:
        texts: List of text regions to mask
        mask_shape: (height, width) of the output mask

    Returns:
        Binary mask array of shape (1, height, width) with 1 where text exists
    """

    mask = Image.new("L", (mask_shape[1], mask_shape[0]), 0)
    draw_mask = ImageDraw.Draw(mask)
    for text in texts:
        draw_mask.polygon([(point.x, point.y) for point in text.bbox], fill=255)

    # slightly dilate the mask to fill gaps between text lines etc.
    mask = mask.filter(ImageFilter.MaxFilter(3))
    # mask = mask.filter(ImageFilter.GaussianBlur)

    # Convert to numpy and normalize to [0, 1] range
    mask_array = np.array(mask).reshape((1, mask_shape[0], mask_shape[1]))
    mask_array = (mask_array > 127).astype(np.uint8)
    return mask_array


def draw_texts(
    image: NDArray[np.float32], texts: list[TranslatedTextItem], debug: bool | None = None
) -> NDArray[np.float32]:
    """
    Draw text onto an image using PIL with rotation support.

    Args:
        image: Normalized float32 image array (values in [0, 1])
        texts: List of text regions with content and styling

    Returns:
        Image array with text drawn, same dtype and range as input
    """

    pil_image = np_img_to_pil(image)

    # Save original mode to restore after drawing
    original_mode = pil_image.mode

    # Convert to RGBA to support transparent text pasting
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")

    pil_image.save("./debug/before_draw.png")

    for text in texts:
        # Calculate rotation angle from bbox
        angle, matrix = get_angle_from_bbox(text.bbox)
        bbox_as_array = [np.array([point.x, point.y]) for point in text.bbox]

        # Get text properties
        ul = text.bbox[0]  # upper-left corner
        text_str = text.translatedText
        color = text.color or "black"

        # Load font
        font: FreeTypeFont | ImageFontType
        try:
            font = ImageFont.truetype(text.detectedFont, float(text.font_size))
        except OSError:
            logger.warning(f"Font '{text.detectedFont}' not found, falling back to default font")
            font = ImageFont.load_default()

        # Get text dimensions from rectified bbox
        rectified_bbox = [point.T @ matrix for point in bbox_as_array]
        text_width = int(rectified_bbox[1][0] - rectified_bbox[0][0])
        text_height = int(rectified_bbox[3][1] - rectified_bbox[0][1])

        # Get PIL's text offset
        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), text_str, font=font)
        text_offset_x, text_offset_y = text_bbox[0], text_bbox[1]

        # Calculate padding needed for rotation
        # Use diagonal length to ensure text fits at any rotation angle
        diagonal = int(np.sqrt(text_width**2 + text_height**2))
        padding = diagonal

        # Create temp image with padding to avoid cutoff during rotation
        temp_size = (text_width + 2 * padding, text_height + 2 * padding)
        temp_text_image = Image.new("RGBA", temp_size, (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_text_image)

        # Draw text at padding position, compensating for PIL's internal offset
        text_pos = (padding - text_offset_x, padding - text_offset_y)
        temp_draw.text(text_pos, text_str, fill=color, font=font)

        if debug:
            unrotated_bbox = temp_draw.textbbox(text_pos, text_str, font=font)
            temp_draw.rectangle(unrotated_bbox, outline="green", width=2)

        # Rotate around top-left (padding, padding) position
        rotated = temp_text_image.rotate(
            -angle,
            center=(padding, padding),
            expand=True,
            resample=Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

        # Calculate paste position: account for canvas expansion during rotation
        # The rotation center shifts by half the size difference (PIL adds symmetric padding)
        shift_x = (rotated.size[0] - temp_size[0]) / 2
        shift_y = (rotated.size[1] - temp_size[1]) / 2
        paste_x = int(ul.x - padding - shift_x)
        paste_y = int(ul.y - padding - shift_y)

        # Paste rotated text onto main image
        pil_image.paste(rotated, (paste_x, paste_y), rotated)
        # print(pil_image.mode, rotated.mode)
        # alpha_composite(pil_image, temp_image)
        if debug:
            pil_draw = ImageDraw.Draw(pil_image)
            bbox_coords = [(point.x, point.y) for point in text.bbox]
            pil_draw.polygon(bbox_coords, outline="red", width=2)
            rotated_width, rotated_height = rotated.size
            pil_text_bbox = [
                (paste_x, paste_y),
                (paste_x + rotated_width, paste_y),
                (paste_x + rotated_width, paste_y + rotated_height),
                (paste_x, paste_y + rotated_height),
            ]
            pil_draw.polygon(pil_text_bbox, outline="blue", width=2)

    # Convert back to original mode to match input format
    if original_mode != "RGBA":
        pil_image = pil_image.convert(original_mode)

    return pil_to_numpy(pil_image)
