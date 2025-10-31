import logging
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

from traenslenzor.image_provider.image_provider import ImageProvider
from traenslenzor.image_renderer.inpainting import Inpainter

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


class ImageRenderer:
    def __init__(self, img_provider: ImageProvider, device="mps") -> None:
        """
        Initialize the ImageRenderer with lazy model loading.

        Args:
            img_provider: ImageProvider instance for handling image I/O. If None, creates default instance.
            device: Device to run the model on (e.g., 'mps', 'cuda', 'cpu')
        """
        self._inpainter: Inpainter | None = None
        self._device = device
        self.img_provider = img_provider

    @property
    def inpainter(self) -> Inpainter:
        """Lazily initialize the inpainter model when first accessed."""
        if self._inpainter is None:
            logger.info("Initializing inpainter model (lazy loading)")
            self._inpainter = Inpainter(device=self._device)
        return self._inpainter

    def create_mask(self, texts: list[Text], mask_shape: tuple[int, int]) -> NDArray[np.uint8]:
        mask = np.zeros(mask_shape, dtype=np.uint8)

        for text in texts:
            x = text["left"]
            y = text["top"]
            w = text["width"]
            h = text["height"]

            y_end = min(y + h, mask_shape[0])
            x_end = min(x + w, mask_shape[1])

            mask[y:y_end, x:x_end] = 1

        return mask.reshape((1, mask_shape[0], mask_shape[1]))

    def draw_texts(self, image: NDArray[np.float32], texts: list[Text]) -> NDArray[np.float32]:
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_draw = ImageDraw.Draw(pil_image)

        for text in texts:
            x = text["left"]
            y = text["top"]
            font_size = text["font_size"]
            color = text["color"]
            font_family = text.get("font_family", "Arial")
            text_str = text["text"]

            try:
                font = ImageFont.truetype(font_family, float(font_size))
            except OSError:
                logger.warning(f"Font '{font_family}' not found, falling back to default font")
                font = ImageFont.load_default()
            pil_draw.text((float(x), float(y)), text_str, fill=color, font=font)
        return (np.array(pil_image) / 255).astype(np.float32)

    def replace_text(
        self,
        image_path: str,
        texts: list[Text],
        inverse_transformation: NDArray[np.float64],
        save_debug: bool = False,
    ) -> PILImage:
        """
        Replace text in an image using inpainting.

        Args:
            image_path: Path to the image file (relative to ImageProvider's image_dir)
            texts: List of text regions to replace
            inverse_transformation: Transformation matrix (currently unused)
            save_debug: If True, save debug images (mask and overlay)

        Returns:
            PIL Image with replaced text
        """
        # Load image using ImageProvider
        rectified_image = self.img_provider.get_image(image_path)

        # Create mask from text regions
        mask = self.create_mask(texts, (rectified_image.height, rectified_image.width))

        # Save debug mask if requested
        if save_debug:
            overlay = Image.fromarray((mask[0] * 255).astype(np.uint8))
            self.img_provider.save_image(overlay, "debug-mask.png")

        # Inpaint the masked regions
        result = self.inpainter.inpaint(rectified_image, mask)

        # Save debug overlay if requested
        if save_debug:
            base = Image.fromarray((result * 255).astype(np.uint8))
            overlay = Image.fromarray((mask[0] * 255).astype(np.uint8))
            debug = ImageProvider.highlight_mask(base, overlay)
            self.img_provider.save_image(debug, "debug-overlay.png")

        # Draw new text on inpainted image
        result = self.draw_texts(result, texts)

        # result = result * inverse_transformation

        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    # Configure logging for the script
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting image inpainting script")

    # Initialize ImageProvider with data directory
    img_provider = ImageProvider("./data")
    renderer = ImageRenderer(img_provider=img_provider)

    # Define text regions to replace
    texts: list[Text] = [
        {
            "text": "Betriebsstörung",
            "left": 571,
            "top": 238,
            "width": 460,
            "height": 80,
            "rotation_in_degrees": 0,
            "font_family": "Arial",
            "color": (0, 0, 0),
            "font_size": 50,
        }
    ]

    # Process image (path is relative to img_provider's image_dir)
    logger.info("Starting inpainting process")
    result = renderer.replace_text(
        "sbahn-betriebsstoerung.png", texts, np.identity(4), save_debug=True
    )

    # Save result
    logger.info("Saving result")
    img_provider.save_image(result, "sbahn-betriebsstoerung-replaced.png")
    logger.info("Done!")
