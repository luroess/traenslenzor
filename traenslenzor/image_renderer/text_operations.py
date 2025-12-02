"""Text rendering and mask creation utilities for image processing."""

import logging
from typing import List

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
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

    return (np.degrees(radians), transformation_matrix)


def create_mask(texts: list[TranslatedTextItem], mask_shape: tuple[int, int]) -> NDArray[np.uint8]:
    """
    Create a binary mask from text regions.

    Args:
        texts: List of text regions to mask
        mask_shape: (height, width) of the output mask

    Returns:
        Binary mask array of shape (1, height, width) with 255 where text exists
    """

    mask = Image.new("L", (mask_shape[1], mask_shape[0]), 0)
    draw_mask = ImageDraw.Draw(mask)
    for text in texts:
        draw_mask.polygon([(point.x, point.y) for point in text.bbox], fill=255)

    return np.array(mask).reshape((1, mask_shape[0], mask_shape[1]))


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

        # Create temp image
        temp_size = (text_width, text_height)
        temp_image = Image.new("RGBA", temp_size, (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_image)

        # Draw text compensating for PIL's internal offset
        text_pos = (-text_offset_x, -text_offset_y)
        temp_draw.text(text_pos, text_str, fill=color, font=font)

        if debug:
            unrotated_bbox = temp_draw.textbbox(text_pos, text_str, font=font)
            temp_draw.rectangle(unrotated_bbox, outline="green", width=2)

        # Rotate around top-left position
        rotated = temp_image.rotate(-angle, center=(0, 0), expand=True, resample=Image.BICUBIC)

        # Calculate paste position: account for canvas expansion during rotation
        # The rotation point shifts by half the size difference (PIL adds symmetric padding)
        shift_x = (rotated.size[0] - temp_size[0]) / 2
        shift_y = (rotated.size[1] - temp_size[1]) / 2
        paste_x = int(ul.x - shift_x)
        paste_y = int(ul.y - shift_y)

        # Paste rotated text onto main image
        pil_image.paste(rotated, (paste_x, paste_y), rotated)
        if debug:
            pil_draw = ImageDraw.Draw(pil_image)
            bbox_coords = [(point.x, point.y) for point in text.bbox]
            pil_draw.polygon(bbox_coords, outline="red", width=2)

    return pil_to_numpy(pil_image)
