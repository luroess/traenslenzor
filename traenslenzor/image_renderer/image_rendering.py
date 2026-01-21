import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

import traenslenzor.image_utils.image_utils as ImageUtils
from traenslenzor.file_server.session_state import RenderReadyItem
from traenslenzor.image_renderer.inpainting import Inpainter
from traenslenzor.image_renderer.text_operations import create_mask, draw_texts

logger = logging.getLogger(__name__)


def save_histogram(data: NDArray[np.float32], filename: str) -> None:
    """Save a histogram of the data to a file."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(data.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()


class ImageRenderer:
    def __init__(self, device: str = "mps") -> None:
        """
        Initialize the ImageRenderer with lazy model loading.

        Args:
            img_provider: ImageProvider instance for handling image I/O. If None, creates default instance.
            device: Device to run the model on (e.g., 'mps', 'cuda', 'cpu')
        """
        self._inpainter: Inpainter | None = None
        self._device: str = device

    @property
    def inpainter(self) -> Inpainter:
        """Lazily initialize the inpainter model when first accessed."""
        if self._inpainter is None:
            logger.info("Initializing inpainter model (lazy loading)")
            self._inpainter = Inpainter(device=self._device)
        return self._inpainter

    async def replace_text(
        self,
        image: PILImage,
        texts: list[RenderReadyItem],
        save_debug: bool = True,
        debug_dir: str = "./debug",
    ) -> PILImage:
        """
        Replace text in an image using inpainting.

        Args:
            image: PIL Image to process
            texts: List of text regions to replace
            inverse_transformation: Transformation matrix (currently unused, optional)
            save_debug: If True, save debug images (mask and overlay)
            debug_dir: Directory to save debug images (default: "./debug")

        Returns:
            PIL Image with replaced text
        """

        # Create mask from text regions
        mask = create_mask(texts, (image.height, image.width))

        # Save debug mask if requested
        if save_debug:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            overlay = Image.fromarray(mask[0] * 255)
            overlay.save(debug_path / "debug-mask.png")

        # Inpaint the masked regions
        result = self.inpainter.inpaint(image, mask)

        # Save debug overlay if requested
        if save_debug:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            overlay = Image.fromarray(mask[0] * 255)
            debug = ImageUtils.highlight_mask(image, overlay, opacity=0.5)
            debug.save(debug_path / "debug-overlay.png")

            Image.fromarray((result.copy() * 255).astype(np.uint8)).save(
                debug_path / "debug-inpainted.png"
            )

        # Draw new text on inpainted image
        result = draw_texts(result, texts, debug=False)

        return ImageUtils.np_img_to_pil(result)

    def transform_image(
        self, image: PILImage, matrix: NDArray[np.float64], original_size: tuple[int, int]
    ) -> PILImage:
        """
        Apply a transformation matrix to the image.

        Args:
            image: PIL Image to transform
            matrix: 3x3 homogeneous transformation matrix

        Returns:
            Transformed PIL Image with transparent background
        """

        # Create mask (255 where image exists)
        mask = np.ones((image.height, image.width), dtype=np.uint8) * 255

        # Transform the mask
        transformed_mask = cv2.warpPerspective(
            mask,
            matrix,
            original_size,
            flags=cv2.INTER_CUBIC,
        )

        # Transform the image
        transformed_image = cv2.warpPerspective(
            np.array(image),
            matrix,
            original_size,
            flags=cv2.INTER_CUBIC,
        )

        # Convert to PIL and apply mask as alpha channel
        img = Image.fromarray(transformed_image).convert("RGBA")
        img.putalpha(Image.fromarray(transformed_mask, mode="L"))

        return img

    def paste_replaced_to_original(self, original, replaced) -> PILImage:
        """
        Paste the replaced image onto the original image, preserving transparency.

        Args:
            original: Original PIL Image
            replaced: Replaced PIL Image with transparency

        Returns:
            Combined PIL Image
        """

        original = original.convert("RGBA")
        resize = replaced.resize(original.size)
        combined = Image.alpha_composite(original, resize)

        return combined.convert("RGB")
