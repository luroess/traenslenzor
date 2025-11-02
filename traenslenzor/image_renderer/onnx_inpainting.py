import logging
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage, Resampling

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"


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
        path_to_model: str = "./traenslenzor/image_renderer/lama/lama_fp32.onnx",
        device: str = "cpu",
    ):
        """Initialize ONNX-based inpainter.

        Args:
            path_to_model: Path to ONNX model file. Downloads if not found.
                Default uses public 512x512 model. Use lama_fp32_1024.onnx for 1024x1024.
            device: Device specification for API compatibility (unused by ONNX).
        """
        logger.info("Initializing ONNX Inpainter")

        try:
            model_path = Path(path_to_model)
            if not model_path.exists():
                logger.info("Downloading LaMa ONNX model from %s", MODEL_URL)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _ = urlretrieve(MODEL_URL, model_path, reporthook=_download_progress)
                logger.info("done downloading")

            logger.debug("Loading LaMa ONNX model from %s", model_path)
            sess_options = onnxruntime.SessionOptions()
            rmodel = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                # providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("LaMa ONNX model loaded successfully")

            # Auto-detect model input size from model metadata
            input_shape = rmodel.get_inputs()[0].shape
            self.size = input_shape[2]  # Assumes shape is [batch, channels, height, width]
            logger.info(f"Model input size: {self.size}x{self.size}")

        except Exception as e:
            logger.error("Failed to load LaMa ONNX model: %s", e, exc_info=True)
            raise

        self.model = rmodel
        self.device = device  # Store for API compatibility

    def preprocess_image(self, image: PILImage) -> tuple[NDArray[np.float32], tuple[int, int]]:
        original_size = image.size
        resized_image = image.resize((self.size, self.size), Resampling.LANCZOS)
        image_array = np.array(resized_image)
        normalized_image = self._normalize_img(image_array)
        padded_image = self._pad_img_to_modulo(normalized_image, 8)
        return (padded_image, original_size)

    def postprocess_image(self, image: PILImage, original_size: tuple[int, int]):
        return image.resize(original_size, Resampling.LANCZOS)

    def inpaint(self, image: PILImage, mask: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Inpaint masked regions of an image.

        Args:
            image: Input PIL Image to inpaint.
            mask: Binary mask array of shape (1, H, W) with uint8 values (0 or 1).

        Returns:
            Inpainted image as float32 numpy array of shape (H, W, 3) with values in [0, 1].
        """
        # Store original dimensions for final output
        original_height, original_width = image.height, image.width

        # Preprocess image
        processed_image, original_size = self.preprocess_image(image.convert("RGB"))
        processed_image = np.expand_dims(processed_image, axis=0)  # (1, C, H, W)

        # Preprocess mask: convert from (1, H, W) to model input format
        # Remove batch dimension temporarily
        mask_2d = mask[0]  # (H, W)

        # Resize mask to match model input size
        mask_pil = Image.fromarray((mask_2d * 255).astype(np.uint8), mode="L")
        mask_resized = mask_pil.resize((self.size, self.size), Resampling.NEAREST)
        mask_array = np.array(mask_resized)

        # Pad and normalize
        mask_padded = self._pad_img_to_modulo(mask_array, 8)
        mask_normalized = mask_padded.astype(np.float32) / 255.0
        mask_model = np.expand_dims(mask_normalized, axis=0)  # (1, H, W)
        mask_model = np.expand_dims(mask_model, axis=0)  # (1, 1, H, W)
        mask_model = (mask_model > 0) * 1

        start = time.time()
        inpainted_image = self.model.run(
            None,
            {
                "image": processed_image.astype(np.float32),
                "mask": mask_model.astype(np.float32),
            },
        )
        end = time.time()
        logger.info("ONNX: Inpainting took %.3f seconds", end - start)

        # Post-process output
        inpainted_image = inpainted_image[0][0]  # Remove batch dimension
        inpainted_image = inpainted_image.transpose(1, 2, 0)  # (H, W, C)

        # Model outputs values in [0, 255] range, convert to uint8 for PIL
        inpainted_uint8 = inpainted_image.clip(0, 255).astype(np.uint8)
        inpainted_pil = Image.fromarray(inpainted_uint8)
        inpainted_resized = inpainted_pil.resize(
            (original_width, original_height), Resampling.LANCZOS
        )

        # Convert back to normalized float32 numpy array [0, 1]
        result = np.array(inpainted_resized).astype(np.float32) / 255.0

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


if __name__ == "__main__":
    inpainter = Inpainter()
    # img_path = Path("./data/sbahn-betriebsstoerung.png")
    # mask_path = Path("./data/sbahn-betriebsstoerung-mask.png")

    # img_path2 = Path("./data/test_onnx_img.jpg")
    # mask_path2 = Path("./data/test_onnx_mask.png")

    img_path3 = Path("./data/sbahn-door-defect.jpg")
    mask_path3 = Path("./data/sbahn-door-defect-mask.png")

    # output_path = Path("./data/sbahn-betriebsstoerung-inpainted_onnx.png")
    # output_path2 = Path("./data/test_onnx_result.png")
    output_path3 = Path("./data/sbahn-door-defect-result.png")

    with Image.open(img_path3) as img, Image.open(mask_path3).convert("L") as mask_pil:
        # Convert mask to numpy array format (1, H, W)
        mask_array = np.array(mask_pil)
        mask_binary = (mask_array > 127).astype(np.uint8)
        mask = mask_binary[np.newaxis, :, :]

        result = inpainter.inpaint(img, mask)

        # Convert result back to PIL Image for saving
        result_uint8 = (result * 255).clip(0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_uint8)
        result_image.save(output_path3)
