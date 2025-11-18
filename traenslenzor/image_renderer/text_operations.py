"""Text rendering and mask creation utilities for image processing."""

import logging
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from PIL.ImageFont import ImageFont as ImageFontType

from traenslenzor.image_utils.image_utils import np_img_to_pil, pil_to_numpy

logger = logging.getLogger(__name__)


class Text(TypedDict):
    text: str
    left: int
    top: int
    width: int
    height: int
    rotation_in_degrees: int
    font_size: int
    color: tuple[int, int, int]
    font_family: str


def create_mask(texts: list[Text], mask_shape: tuple[int, int]) -> NDArray[np.uint8]:
    """
    Create a binary mask from text regions.

    Args:
        texts: List of text regions to mask
        mask_shape: (height, width) of the output mask

    Returns:
        Binary mask array of shape (1, height, width) with 255 where text exists
    """
    mask = np.zeros(mask_shape, dtype=np.uint8)

    for text in texts:
        x = text["left"]
        y = text["top"]
        w = text["width"]
        h = text["height"]

        y_end = min(y + h, mask_shape[0])
        x_end = min(x + w, mask_shape[1])

        mask[y:y_end, x:x_end] = 255

    return mask.reshape((1, mask_shape[0], mask_shape[1]))


def draw_texts(image: NDArray[np.float32], texts: list[Text]) -> NDArray[np.float32]:
    """
    Draw text onto an image using PIL.

    Args:
        image: Normalized float32 image array (values in [0, 1])
        texts: List of text regions with content and styling

    Returns:
        Image array with text drawn, same dtype and range as input
    """
    pil_image = np_img_to_pil(image)
    pil_draw = ImageDraw.Draw(pil_image)

    for text in texts:
        x = text["left"]
        y = text["top"]
        font_size = text["font_size"]
        color = text["color"]
        font_family = text.get("font_family", "Arial")
        text_str = text["text"]

        font: FreeTypeFont | ImageFontType
        try:
            font = ImageFont.truetype(font_family, float(font_size))
        except OSError:
            logger.warning(f"Font '{font_family}' not found, falling back to default font")
            font = ImageFont.load_default()
        pil_draw.text((float(x), float(y)), text_str, fill=color, font=font)
    return pil_to_numpy(pil_image)
