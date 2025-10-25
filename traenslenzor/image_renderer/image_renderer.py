from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage, Resampling

from traenslenzor.image_renderer.utils import norm_img, pad_img_to_modulo


class ImageRenderer:
    def __init__(self) -> None:
        self.lama: torch.jit.ScriptModule = torch.jit.load(
            "./traenslenzor/image_renderer/lama/big-lama.pt", map_location="cpu"
        )
        self.lama.eval()
        self.debug = True

    def inpaint_mask(self, img: PILImage, mask_in: PILImage) -> PILImage:
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        if img.mode != "RGB":
            if self.debug:
                print("converting img to RGB")
            img = img.convert("RGB")

        if mask_in.mode != "L":
            if self.debug:
                print("converting mask to L")
            mask_in = mask_in.convert("L")

        if mask_in.size != img.size:
            if self.debug:
                print("img and mask size differ")
            mask_in = mask_in.resize(img.size, Resampling.NEAREST)

        image = norm_img(np.array(img))
        mask = norm_img(np.array(mask_in))

        if self.debug:
            print(f"Image shape after norm_img: {image.shape}")
            print(f"Mask shape after norm_img: {mask.shape}")

        # Pad to be divisible by 8
        image = pad_img_to_modulo(image, 8)
        mask = pad_img_to_modulo(mask, 8)

        if self.debug:
            print(f"Image shape after padding: {image.shape}")
            print(f"Mask shape after padding: {mask.shape}")

            print(f"Mask unique values before thresholding: {np.unique(mask)}")
        mask = (mask > 0.5) * 1
        if self.debug:
            print(f"Mask unique values after thresholding: {np.unique(mask)}")
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.debug:
            print(f"Image tensor shape: {image.shape}")
            print(f"Mask tensor shape: {mask.shape}")

        inpainted_image: torch.Tensor = self.lama(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).cpu().detach().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        # cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return Image.fromarray(cur_res)


def diagnose_png(path):
    with Image.open(path).convert("L") as img:
        print(f"\n=== {path} ===")
        print(f"Mode: {img.mode}")
        print(f"Size: {img.size}")
        print(f"Format: {img.format}")
        # print(f"Info: {img.info}")

        # Check for transparency
        if img.mode == "RGBA":
            alpha = img.split()[3]
            alpha_arr = np.array(alpha)
            print(f"Has alpha channel: min={alpha_arr.min()}, max={alpha_arr.max()}")
            print(f"Has transparency: {alpha_arr.min() < 255}")

        # Get pixel value range
        arr = np.array(img)
        print(f"Array shape: {arr.shape}")
        print(f"Array dtype: {arr.dtype}")
        print(f"Pixel value range: min={arr.min()}, max={arr.max()}")
        print(f"Mean: {arr.mean()}")


if __name__ == "__main__":
    img_path = Path("./data/sbahn-betriebsstoerung.png")
    img_path_2 = Path("./data/image_1.png")

    # diagnose_png(img_path)
    # diagnose_png(img_path_2)
    mask_path = Path("./data/sbahn-betriebsstoerung-mask.png")
    mask_path2 = Path("./data/mask_1.png")

    diagnose_png(mask_path)
    diagnose_png(mask_path2)

    renderer = ImageRenderer()

    with Image.open(img_path) as img, Image.open(mask_path).convert("L") as mask:
        img_array = np.array(img)
        # mask2 = create_box_mask(0, 0, 800, 40, (img_array.shape[0], img_array.shape[1]))
        # mask2 = Image.fromarray(mask2)
        print("\n\ngoing to inpaint")
        in_painted = renderer.inpaint_mask(img, mask)
        print("inpainting done")

        # result = Image.fromarray(in_painted)

        print("showing result")
        # in_painted.show()
        in_painted.save(Path("./data/sbahn-betriebsstoerung-inpainted.png"))
