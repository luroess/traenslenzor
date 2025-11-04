import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from PIL.ImageFont import FreeTypeFont, ImageFont as ImageFontType

import traenslenzor.image_utils.image_utils as ImageUtils
from traenslenzor.file_server.client import FileClient
from traenslenzor.image_renderer.inpainting import Inpainter

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


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

    def create_mask(self, texts: list[Text], mask_shape: tuple[int, int]) -> NDArray[np.uint8]:
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

            font: FreeTypeFont | ImageFontType
            try:
                font = ImageFont.truetype(font_family, float(font_size))
            except OSError:
                logger.warning(f"Font '{font_family}' not found, falling back to default font")
                font = ImageFont.load_default()
            pil_draw.text((float(x), float(y)), text_str, fill=color, font=font)
        return (np.array(pil_image) / 255).astype(np.float32)

    async def replace_text(
        self,
        image_id: str,
        texts: list[Text],
        inverse_transformation: NDArray[np.float64],
        save_debug: bool = False,
        debug_dir: str = "./debug",
    ) -> PILImage:
        """
        Replace text in an image using inpainting.

        Args:
            image_id: Id of the image file (from our FileClient)
            texts: List of text regions to replace
            inverse_transformation: Transformation matrix (currently unused)
            save_debug: If True, save debug images (mask and overlay)
            debug_dir: Directory to save debug images (default: "./debug")

        Returns:
            PIL Image with replaced text
        """
        # Load image using ImageProvider
        rectified_image = await FileClient.get_image(image_id)

        if rectified_image is None:
            raise FileNotFoundError(f"Image with ID '{image_id}' not found.")

        # Create mask from text regions
        mask = self.create_mask(texts, (rectified_image.height, rectified_image.width))

        # Save debug mask if requested
        if save_debug:
            debug_path = Path(debug_dir)
            mask_dir = debug_path / "debug"
            mask_dir.mkdir(parents=True, exist_ok=True)
            overlay = Image.fromarray(mask[0])
            overlay.save(mask_dir / "debug-mask.png")

        # Inpaint the masked regions
        result = self.inpainter.inpaint(rectified_image, mask)

        # Save debug overlay if requested
        if save_debug:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            base = Image.fromarray((result * 255).astype(np.uint8))
            overlay = Image.fromarray(mask[0])
            debug = ImageUtils.highlight_mask(base, overlay)
            debug.save(debug_path / "debug-overlay.png")

        # Draw new text on inpainted image
        result = self.draw_texts(result, texts)

        # result = result * inverse_transformation

        return ImageUtils.np_img_to_pil(result)
