import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traenslenzor.image_provider.image_provider import ImageProvider


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def provider(temp_dir: str) -> ImageProvider:
    return ImageProvider(temp_dir)


@pytest.fixture
def sample_image(temp_dir: str) -> str:
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    path = Path(temp_dir) / "test.png"
    img.save(path)
    return "test.png"


def test_init_creates_image_dir_path(provider: ImageProvider, temp_dir: str) -> None:
    assert provider.image_dir == Path(temp_dir)


def test_get_image_returns_pil_image(provider: ImageProvider, sample_image: str) -> None:
    img = provider.get_image(sample_image)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)


def test_get_image_nonexistent_file_raises_error(provider: ImageProvider) -> None:
    with pytest.raises(FileNotFoundError):
        provider.get_image("nonexistent.png")


def test_get_image_as_numpy_returns_normalized_array(
    provider: ImageProvider, sample_image: str
) -> None:
    arr = provider.get_image_as_numpy(sample_image)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0
    assert arr.min() >= 0.0


def test_save_image_creates_file(provider: ImageProvider) -> None:
    img = Image.new("RGB", (50, 50))
    provider.save_image(img, "output.png")
    assert (provider.image_dir / "output.png").exists()


def test_save_image_creates_parent_directories(provider: ImageProvider) -> None:
    img = Image.new("RGB", (50, 50))
    provider.save_image(img, "subdir/output.png")
    assert (provider.image_dir / "subdir" / "output.png").exists()


def test_save_image_preserves_content(provider: ImageProvider) -> None:
    img = Image.new("RGB", (50, 50), color=(200, 100, 50))
    provider.save_image(img, "output.png")
    loaded = provider.get_image("output.png")
    assert loaded.size == (50, 50)


def test_img_to_pil_converts_normalized_array() -> None:
    arr = np.random.rand(100, 100, 3).astype(np.float32)
    img = ImageProvider.img_to_pil(arr)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)


def test_img_to_pil_handles_out_of_range_values() -> None:
    arr = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
    img = ImageProvider.img_to_pil(arr)
    pixels = list(img.getdata())
    assert pixels[0] == (255, 0, 127)


def test_highlight_mask_applies_red_overlay() -> None:
    base = Image.new("RGB", (100, 100), color=(255, 255, 255))  # white
    mask = Image.new("L", (100, 100), color=255)  # full mask
    result = ImageProvider.highlight_mask(base, mask)

    # Check center pixel has red overlay applied
    center_pixel: tuple[int, int, int] = result.getpixel((50, 50))  # pyright: ignore[reportAssignmentType]
    assert center_pixel[0] > center_pixel[1]  # R > G (more red than green)
    assert center_pixel[0] > center_pixel[2]  # R > B (more red than blue)


def test_highlight_mask_respects_opacity() -> None:
    # Use gray base so all channels can demonstrate opacity effect
    base = Image.new("RGB", (100, 100), color=(128, 128, 128))
    mask = Image.new("L", (100, 100), color=255)

    result_low = ImageProvider.highlight_mask(base.copy(), mask, opacity=0.3)
    result_high = ImageProvider.highlight_mask(base.copy(), mask, opacity=0.9)

    pixel_low: tuple[int, int, int] = result_low.getpixel((50, 50))  # pyright: ignore[reportAssignmentType]
    pixel_high: tuple[int, int, int] = result_high.getpixel((50, 50))  # pyright: ignore[reportAssignmentType]

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

    result = ImageProvider.highlight_mask(base, mask)

    # Check dimensions are preserved
    assert result.size == (200, 150)

    # Inside masked region: should have red overlay
    inside_pixel: tuple[int, int, int] = result.getpixel((100, 75))  # pyright: ignore[reportAssignmentType]
    assert inside_pixel[0] > inside_pixel[1]  # Red component dominant

    # Outside masked region: should remain white
    outside_pixel = result.getpixel((10, 10))
    assert outside_pixel == (255, 255, 255)
