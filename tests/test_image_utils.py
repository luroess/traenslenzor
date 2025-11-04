import numpy as np
import pytest
from PIL import Image

import traenslenzor.image_utils.image_utils as ImageUtils
from traenslenzor.file_server.client import FileClient


@pytest.fixture
async def sample_image_id(file_server: FileClient) -> str:
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    id = await file_server.put_img("test.png", img)

    if id is None:
        raise RuntimeError("Failed to upload sample image to file server")
    return id


@pytest.mark.anyio
async def test_get_image_as_numpy_returns_normalized_array(sample_image_id: str) -> None:
    arr = await FileClient.get_image_as_numpy(sample_image_id)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0
    assert arr.min() >= 0.0


def test_img_to_pil_converts_normalized_array() -> None:
    arr = np.random.rand(100, 100, 3).astype(np.float32)
    img = ImageUtils.np_img_to_pil(arr)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)


def test_img_to_pil_handles_out_of_range_values() -> None:
    arr = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
    img = ImageUtils.np_img_to_pil(arr)
    pixels = list(img.getdata())  # pyright: ignore[reportArgumentType]
    assert pixels[0] == (255, 0, 127)


def test_highlight_mask_applies_red_overlay() -> None:
    base = Image.new("RGB", (100, 100), color=(255, 255, 255))  # white
    mask = Image.new("L", (100, 100), color=255)  # full mask
    result = ImageUtils.highlight_mask(base, mask)

    # Check center pixel has red overlay applied
    center_pixel: tuple[int, int, int] = result.getpixel((50, 50))  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert center_pixel[0] > center_pixel[1]  # R > G (more red than green)
    assert center_pixel[0] > center_pixel[2]  # R > B (more red than blue)


def test_highlight_mask_respects_opacity() -> None:
    # Use gray base so all channels can demonstrate opacity effect
    base = Image.new("RGB", (100, 100), color=(128, 128, 128))
    mask = Image.new("L", (100, 100), color=255)

    result_low = ImageUtils.highlight_mask(base.copy(), mask, opacity=0.3)
    result_high = ImageUtils.highlight_mask(base.copy(), mask, opacity=0.9)

    pixel_low: tuple[int, int, int] = result_low.getpixel((50, 50))  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    pixel_high: tuple[int, int, int] = result_high.getpixel((50, 50))  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # Higher opacity = more red overlay influence
    # Alpha blending: result = base * (1-alpha) + overlay * alpha
    assert pixel_high[0] > pixel_low[0]  # More red at high opacity
    assert pixel_high[1] < pixel_low[1]  # Less green at high opacity
    assert pixel_high[2] < pixel_low[2]  # Less blue at high opacity


def test_highlight_mask_preserves_base_dimensions() -> None:
    base = Image.new("RGB", (200, 150), color=(255, 255, 255))
    # Create partial mask: only center region is masked
    mask = Image.new("L", (200, 150), color=0)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(mask)
    draw.rectangle([50, 50, 150, 100], fill=255)

    result = ImageUtils.highlight_mask(base, mask)

    # Check dimensions are preserved
    assert result.size == (200, 150)

    # Inside masked region: should have red overlay
    inside_pixel: tuple[int, int, int] = result.getpixel((100, 75))  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert inside_pixel[0] > inside_pixel[1]  # Red component dominant

    # Outside masked region: should remain white
    outside_pixel = result.getpixel((10, 10))
    assert outside_pixel == (255, 255, 255)
