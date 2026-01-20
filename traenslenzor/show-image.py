import asyncio

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from traenslenzor.file_server.client import FileClient

MAX_DISPLAY_PIXELS = 50_000_000


async def fetch_image_pil(
    file_id: str,
    timeout: float = 60.0,
) -> Image.Image:
    """
    Fetches an image from the file server and returns it as a PIL Image.
    """
    image_np = await FileClient.get_image(
        file_id,
        timeout=timeout,
        max_pixels=MAX_DISPLAY_PIXELS,
    )
    if image_np is None:
        raise ValueError(f"Could not fetch image {file_id}")
    return Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np


async def main():
    deskewed_image = await fetch_image_pil("b59e477b-b1e5-45a8-9c19-036ef2bc9ead")

    width, height = deskewed_image.size
    print(f"Deskewed image size: {width}x{height}")

    super_res = await fetch_image_pil("6b1ac46a-e5b0-4a5d-8519-402465ffda5b")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(deskewed_image)
    ax1.set_title(f"Deskewed Image ({width}x{height})")
    ax1.axis("off")
    ax2.imshow(super_res)
    ax2.set_title("Super-Resolved Image")
    ax2.axis("off")
    plt.show()


if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(main())
    else:
        # Likely running in a notebook or interactive loop.
        asyncio.create_task(main())
