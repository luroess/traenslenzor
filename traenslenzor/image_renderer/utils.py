import numpy as np
from numpy.typing import NDArray


def norm_img(np_img: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def pad_img_to_modulo(img: NDArray[np.float64], mod: int) -> NDArray[np.float64]:
    """Pad image to make dimensions divisible by mod"""
    channels, height, width = img.shape
    out_height = (height + mod - 1) // mod * mod
    out_width = (width + mod - 1) // mod * mod

    pad_h = out_height - height
    pad_w = out_width - width

    return np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
