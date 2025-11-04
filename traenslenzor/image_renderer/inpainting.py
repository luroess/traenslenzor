import logging
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import Image as PILImage

import traenslenzor.image_utils.image_utils as ImageUtils

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100.0 / total_size, 100.0)
        logger.info(
            "Download progress: %.1f%% (%d/%d bytes)",
            percent,
            downloaded,
            total_size,
        )


class Inpainter:
    def __init__(
        self,
        path_to_model: str = "./traenslenzor/image_renderer/lama/big-lama.pt",
        device: str = "mps",
    ) -> None:
        logger.info("Initializing ImageRenderer")

        self.device = torch.device(device)

        try:
            model_path = Path(path_to_model)
            if not model_path.exists():
                logger.info("Downloading LaMa model from %s", MODEL_URL)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _ = urlretrieve(MODEL_URL, model_path, reporthook=_download_progress)
                logger.info(
                    "done downloading",
                )

            logger.debug("Loading LaMa model from %s", model_path)
            self.LaMa: torch.jit.ScriptModule = torch.jit.load(model_path, map_location=self.device)

            self.LaMa.to(self.device)
            self.LaMa.eval()
            logger.info("LaMa model loaded successfully")
        except Exception as e:
            logger.error("Failed to load LaMa model: %s", e, exc_info=True)
            raise

    def inpaint(self, img: PILImage, mask_in: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Inpaint an image using the provided mask.

        Input image and output image have same size.

        Args:
            img: [H, W, C] RGB PIL Image
            mask_in: [H, W] L mode PIL Image

        Returns:
            Inpainted RGB PIL Image
        """
        logger.debug("Starting inpaint_mask with img size=%s, mask size=%s", img.size, mask_in.size)

        # Convert image to RGB if needed
        if img.mode != "RGB":
            logger.debug("Converting image from %s to RGB", img.mode)
            img = img.convert("RGB")

        # Store original dimensions for cropping after inpainting
        img_array = np.array(img)
        original_height, original_width = img_array.shape[:2]

        # Normalize images
        image = self._normalize_img(img_array)

        # Pad to be divisible by 8
        padded_image = self._pad_img_to_modulo(image, 8)
        padded_mask = self._pad_img_to_modulo(mask_in, 8)
        logger.debug(
            "After padding: image shape=%s, mask shape=%s", padded_image.shape, padded_mask.shape
        )

        # Convert to tensors and add batch dimension
        image_tensor = torch.from_numpy(padded_image).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0).to(self.device)
        logger.debug("Tensor shapes: image=%s, mask=%s", image_tensor.shape, mask_tensor.shape)

        # Run inpainting
        logger.info("Running LaMa inpainting")
        try:
            with torch.inference_mode():
                start = time.time()
                inpainted_images: torch.Tensor = self.LaMa(image_tensor, mask_tensor)
                end = time.time()
            logger.info(f"JIT: Inpainting took {end - start:.3f} seconds")
        except Exception as e:
            logger.error("Inpainting failed: %s", e, exc_info=True)
            raise

        # Post-process result
        result = inpainted_images[0].permute(1, 2, 0).cpu().detach().numpy()

        # Crop back to original dimensions
        result = result[:original_height, :original_width, :]
        logger.debug("Cropped result to original size: %s", result.shape)

        return result

    def _normalize_img(self, np_img: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(np_img.shape) == 2:
            np_img = np_img[:, :, np.newaxis]
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = (np_img / 255).astype(np.float32)
        return np_img

    def _pad_img_to_modulo(
        self, np_img: NDArray[np.generic], mod: int, mode: str = "edge"
    ) -> NDArray[np.generic]:
        """Pad image to make dimensions divisible by mod"""
        if len(np_img.shape) == 2:
            height = int(np_img.shape[0])
            width = int(np_img.shape[1])
            out_height = (height + mod - 1) // mod * mod
            out_width = (width + mod - 1) // mod * mod
            pad_h = out_height - height
            pad_w = out_width - width
            return np.pad(np_img, ((0, pad_h), (0, pad_w)), mode=mode)  # type: ignore[no-any-return,call-overload]
        elif len(np_img.shape) == 3:
            height = int(np_img.shape[1])
            width = int(np_img.shape[2])
            out_height = (height + mod - 1) // mod * mod
            out_width = (width + mod - 1) // mod * mod
            pad_h = out_height - height
            pad_w = out_width - width
            return np.pad(np_img, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)  # type: ignore[no-any-return,call-overload]
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {np_img.shape}")
