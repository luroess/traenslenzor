import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

import traenslenzor.image_utils.image_utils as ImageUtils
from traenslenzor.file_server.session_state import TranslatedTextItem
from traenslenzor.image_renderer.inpainting import Inpainter
from traenslenzor.image_renderer.text_operations import create_mask, draw_texts

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


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
        texts: list[TranslatedTextItem],
        inverse_transformation: NDArray[np.float64] | None = None,
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
        print(mask)

        # Save debug mask if requested
        if save_debug:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            overlay = Image.fromarray(mask[0])
            overlay.save(debug_path / "debug-mask.png")

        # Inpaint the masked regions
        result = self.inpainter.inpaint(image, mask)

        # Save debug overlay if requested
        if save_debug:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            base = Image.fromarray((result * 255).astype(np.uint8))
            overlay = Image.fromarray(mask[0])
            debug = ImageUtils.highlight_mask(base, overlay)
            debug.save(debug_path / "debug-overlay.png")

        # Draw new text on inpainted image
        result = draw_texts(result, texts)

        # result = result * inverse_transformation

        return ImageUtils.np_img_to_pil(result)
