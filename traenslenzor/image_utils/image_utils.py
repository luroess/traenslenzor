import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage


def pil_to_numpy(pil_img: PILImage) -> NDArray[np.float32]:
    """Convert PIL Image back to normalized numpy array"""
    return np.array(pil_img, dtype=np.float32) / 255.0


def np_img_to_pil(np_img: NDArray[np.float32]) -> PILImage:
    """Convert normalized numpy array back to PIL Image"""
    return Image.fromarray(np.clip(np_img * 255, 0, 255).astype(np.uint8))


def highlight_mask(base_image: PILImage, mask: PILImage, opacity=0.5) -> PILImage:
    """
    Overlay an image with custom opacity (0.0 to 1.0)
    """

    # Adjust the opacity of the overlay
    overlay = Image.new("RGB", (mask.width, mask.height), (255, 0, 0))
    mask = mask.point(lambda p: int(p * opacity))  # Scale alpha values

    # Paste with transparency
    base_image.paste(overlay, (0, 0), mask)
    return base_image
