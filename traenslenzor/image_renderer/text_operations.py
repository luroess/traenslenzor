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


def draw_texts(image: NDArray[np.float32], texts: list[TranslatedTextItem]) -> NDArray[np.float32]:
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
        ll = text.bbox[3]  # lower-left corner
        font_size = text.font_size
        color = text.color or "black"
        font_family = text.detectedFont
        text_str = text.translatedText

        # Load font
        font: FreeTypeFont | ImageFontType
        try:
            font = ImageFont.truetype(font_family, float(font_size))
        except OSError:
            logger.warning(f"Font '{font_family}' not found, falling back to default font")
            font = ImageFont.load_default()

        # Get text dimensions from rectified bbox
        rectified_bbox = [point.T @ matrix for point in bbox_as_array]
        text_width = int(rectified_bbox[1][0] - rectified_bbox[0][0])
        text_height = int(rectified_bbox[3][1] - rectified_bbox[0][1])

        # Get actual text bounding box to find PIL's offset
        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), text_str, font=font)
        text_offset_x = text_bbox[0]  # Left offset
        text_offset_y = text_bbox[1]  # Top offset

        # Create temp image with padding to avoid cutoff during rotation
        padding = max(text_width, text_height)
        temp_width = text_width + 2 * padding
        temp_height = text_height + 2 * padding
        temp_image = Image.new("RGBA", (temp_width, temp_height), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_image)

        # Draw text at padding position, compensating for PIL's internal offset
        text_x = padding - text_offset_x
        text_y = padding - text_offset_y
        temp_draw.text((text_x, text_y), text_str, fill=color, font=font)

        # Draw bounding box around text in temp image (before rotation)
        unrotated_bbox = temp_draw.textbbox((text_x, text_y), text_str, font=font)
        temp_draw.rectangle(unrotated_bbox, outline="green", width=2)

        # Rotate around top-left (padding, padding) position
        # This keeps the text area's top-left fixed during rotation
        rotated = temp_image.rotate(
            -angle, center=(padding, padding), expand=True, resample=Image.BICUBIC
        )

        # After rotation with expand, calculate how much the center shifted
        # The old center was at (temp_width/2, temp_height/2)
        # The rotation center (padding, padding) stays fixed
        # So we need to find where (padding, padding) is in the expanded image

        old_center_x = temp_width / 2
        old_center_y = temp_height / 2
        new_center_x = rotated.size[0] / 2
        new_center_y = rotated.size[1] / 2

        # The rotation center in the new image
        # Since we rotated around (padding, padding) with expand, that point stays conceptually fixed
        # but the canvas expanded, so we need to account for the shift
        center_shift_x = new_center_x - old_center_x
        center_shift_y = new_center_y - old_center_y

        # Where (padding, padding) ended up in the rotated image
        rotated_padding_x = padding + center_shift_x
        rotated_padding_y = padding + center_shift_y

        # Paste position: we want (rotated_padding_x, rotated_padding_y) to align with (ul.x, ul.y)
        paste_x = int(ul.x - rotated_padding_x)
        paste_y = int(ul.y - rotated_padding_y)

        # Paste rotated text onto main image
        pil_image.paste(rotated, (paste_x, paste_y), rotated)

        # Calculate and log rotated text box coordinates in original image space
        rotated_width, rotated_height = rotated.size
        pil_text_bbox = [
            (paste_x, paste_y),  # top-left
            (paste_x + rotated_width, paste_y),  # top-right
            (paste_x + rotated_width, paste_y + rotated_height),  # bottom-right
            (paste_x, paste_y + rotated_height),  # bottom-left
        ]

        # Draw the original bounding box for visualization
        pil_draw = ImageDraw.Draw(pil_image)
        bbox_coords = [(point.x, point.y) for point in text.bbox]
        pil_draw.polygon(bbox_coords, outline="red", width=2)

        # Draw PIL's rotated text bounding box
        pil_draw.polygon(pil_text_bbox, outline="blue", width=2)

    return pil_to_numpy(pil_image)
