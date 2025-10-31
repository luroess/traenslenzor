import time
from pathlib import Path

import numpy as np
import onnxruntime
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage, Resampling


class Inpainter:
    def __init__(self):
        print("Available providers:")
        for provider in onnxruntime.get_available_providers():
            print(f"  - {provider}")

        sess_options = onnxruntime.SessionOptions()
        rmodel = onnxruntime.InferenceSession(
            "traenslenzor/image_renderer/lama/lama_fp32_1024.onnx",
            sess_options=sess_options,
            # providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

        self.size = 1024
        self.model = rmodel

    def preprocess_image(self, image: PILImage):
        original_size = image.size
        image = image.resize((self.size, self.size), Resampling.LANCZOS)
        image = np.array(image)
        image = self._normalize_img(image)
        image = self._pad_img_to_modulo(image, 8)
        return (image, original_size)

    def postprocess_image(self, image: PILImage, original_size: tuple[int, int]):
        return image.resize(original_size, Resampling.LANCZOS)

    def inpaint(self, image: PILImage, mask: PILImage):
        image, original_size = self.preprocess_image(image.convert("RGB"))
        image = np.expand_dims(image, axis=0)  # (C, H, W)
        print("image shape")
        print(image.shape)
        mask = mask.resize((self.size, self.size))

        mask = np.expand_dims(mask, axis=0)  # (C, H, W)
        mask = self._pad_img_to_modulo(np.array(mask), 8)
        mask = mask.astype(np.float32) / 255.0
        # mask = np.transpose(mask, (2, 0, 1))  # (C, H, W)
        mask = np.expand_dims(mask, axis=0)  # (1, C, H, W)
        mask = (mask > 0) * 1

        start = time.time()
        # Placeholder for inpainting logic
        inpainted_image = self.model.run(
            None,
            {
                "image": np.array(image).astype(np.float32),
                "mask": np.array(mask).astype(np.float32),
            },
        )
        end = time.time()
        print(f"ONNX: Inpainting took {end - start:.3f} seconds")
        inpainted_image = inpainted_image[0][0]
        inpainted_image = inpainted_image.transpose(1, 2, 0)
        # inpainted_image = np.clip(inpainted_image * 255, 0, 255)
        inpainted_image = inpainted_image.astype(np.uint8)

        inpainted_image = Image.fromarray(inpainted_image)
        inpainted_image = self.postprocess_image(inpainted_image, original_size)

        return inpainted_image

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
            height, width = np_img.shape
            out_height = (height + mod - 1) // mod * mod
            out_width = (width + mod - 1) // mod * mod
            pad_h = out_height - height
            pad_w = out_width - width
            return np.pad(np_img, ((0, pad_h), (0, pad_w)), mode=mode)
        elif len(np_img.shape) == 3:
            channels, height, width = np_img.shape
            out_height = (height + mod - 1) // mod * mod
            out_width = (width + mod - 1) // mod * mod
            pad_h = out_height - height
            pad_w = out_width - width
            return np.pad(np_img, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {np_img.shape}")


if __name__ == "__main__":
    inpainter = Inpainter()
    img_path = Path("./data/sbahn-betriebsstoerung.png")
    mask_path = Path("./data/sbahn-betriebsstoerung-mask.png")

    img_path2 = Path("./data/test_onnx_img.jpg")
    mask_path2 = Path("./data/test_onnx_mask.png")

    img_path3 = Path("./data/sbahn-door-defect.jpg")
    mask_path3 = Path("./data/sbahn-door-defect-mask.png")

    output_path = Path("./data/sbahn-betriebsstoerung-inpainted_onnx.png")
    output_path2 = Path("./data/test_onnx_result.png")
    output_path3 = Path("./data/sbahn-door-defect-result.png")

    with Image.open(img_path3) as img, Image.open(mask_path3).convert("L") as mask:
        result = inpainter.inpaint(img, mask)
        result.save(output_path3)
