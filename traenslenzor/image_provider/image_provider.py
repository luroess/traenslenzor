from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage


class ImageProvider:
    def __init__(self, image_dir: str = "./data") -> None:
        self.image_dir = Path(image_dir)

    def get_image(self, image: str) -> PILImage:
        return Image.open(self.image_dir / image)

    def get_image_as_numpy(self, image: str) -> NDArray[np.float32]:
        pil_image = self.get_image(image)
        np_image = np.array(pil_image)
        np_image = (np_image / 255).astype(np.float32)
        return np_image

    def save_image(self, image: PILImage, filename: str) -> None:
        """Save an image to the image directory"""
        output_path = self.image_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    @staticmethod
    def img_to_pil(np_img: NDArray[np.float32]) -> PILImage:
        """Convert normalized numpy array back to PIL Image"""
        return Image.fromarray(np.clip(np_img * 255, 0, 255).astype(np.uint8))

    @staticmethod
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
