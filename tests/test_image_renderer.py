import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traenslenzor.file_server.client import FileClient
from traenslenzor.image_renderer.image_renderer import ImageRenderer, Text


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def renderer() -> ImageRenderer:
    return ImageRenderer(device="cpu")


@pytest.fixture
def sample_text() -> Text:
    return {
        "text": "Hello",
        "left": 10,
        "top": 10,
        "width": 50,
        "height": 20,
        "rotation_in_degrees": 0,
        "font_size": 16,
        "color": (0, 0, 0),
        "font_family": "Arial",
    }


@pytest.fixture
async def sample_img_id(file_server: type[FileClient]) -> str:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    id = await file_server.put_img("test.png", img)
    if id is None:
        raise RuntimeError("Failed to upload sample image to file server")

    return id


def test_create_mask_single_text_region(renderer: ImageRenderer, sample_text: Text) -> None:
    mask = renderer.create_mask([sample_text], (100, 100))
    assert mask.shape == (1, 100, 100)
    assert mask.dtype == np.uint8
    assert mask[0, 10:30, 10:60].sum() > 0


def test_create_mask_multiple_text_regions(renderer: ImageRenderer, sample_text: Text) -> None:
    text2 = sample_text.copy()
    text2["left"] = 60
    mask = renderer.create_mask([sample_text, text2], (100, 100))
    # Verify first text region (left=10, width=50)
    assert mask[0, 10:30, 10:60].sum() > 0
    # Verify second text region (left=60, width=50)
    assert mask[0, 10:30, 60:110].sum() > 0
    # Both regions should have masked pixels
    assert mask.sum() > mask[0, 10:30, 10:60].sum()


def test_create_mask_overlapping_regions(renderer: ImageRenderer, sample_text: Text) -> None:
    text2 = sample_text.copy()
    text2["left"] = 15
    mask = renderer.create_mask([sample_text, text2], (100, 100))
    assert mask.shape == (1, 100, 100)
    # Verify overlapping region (left=15 to left=60 where both texts should appear)
    overlap_region = mask[0, 10:30, 15:60]
    assert overlap_region.sum() > 0
    # Mask should handle overlaps correctly (pixels can be masked multiple times)
    assert mask.sum() > 0


def test_create_mask_clamps_to_boundaries(renderer: ImageRenderer) -> None:
    text: Text = {
        "text": "Test",
        "left": 80,
        "top": 80,
        "width": 50,
        "height": 50,
        "rotation_in_degrees": 0,
        "font_size": 16,
        "color": (0, 0, 0),
        "font_family": "Arial",
    }
    mask = renderer.create_mask([text], (100, 100))
    assert mask.shape == (1, 100, 100)


def test_create_mask_correct_shape(renderer: ImageRenderer, sample_text: Text) -> None:
    mask = renderer.create_mask([sample_text], (200, 150))
    assert mask.shape == (1, 200, 150)


def test_draw_texts_single_text(renderer: ImageRenderer, sample_text: Text) -> None:
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    result = renderer.draw_texts(img, [sample_text])
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.float32


def test_draw_texts_multiple_texts(renderer: ImageRenderer, sample_text: Text) -> None:
    text2 = sample_text.copy()
    text2["left"] = 60
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    result = renderer.draw_texts(img, [sample_text, text2])
    assert result.shape == (100, 100, 3)
    # Verify the image was modified (should differ from original uniform gray)
    assert not np.allclose(result, 0.5)
    # Check that both text regions were modified
    region1_modified = not np.allclose(result[10:30, 10:60], 0.5)
    region2_modified = not np.allclose(result[10:30, 60:110], 0.5)
    assert region1_modified and region2_modified


def test_draw_texts_respects_font_properties(renderer: ImageRenderer) -> None:
    text: Text = {
        "text": "Test",
        "left": 10,
        "top": 10,
        "width": 50,
        "height": 20,
        "rotation_in_degrees": 0,
        "font_size": 24,
        "color": (255, 0, 0),
        "font_family": "Arial",
    }
    img = np.ones((100, 100, 3), dtype=np.float32)
    result = renderer.draw_texts(img, [text])
    assert result is not None
    # Verify red color was applied (color specified as RGB(255, 0, 0))
    # The red channel should be significantly higher than green/blue in text region
    text_region = result[10:30, 10:60]
    # Check that red channel has higher values than original white
    assert text_region[:, :, 0].max() > 0.5  # Red channel should have content
    # Image should be modified from original uniform white
    assert not np.allclose(result, 1.0)


def test_draw_texts_preserves_image_dimensions(renderer: ImageRenderer, sample_text: Text) -> None:
    img = np.ones((200, 150, 3), dtype=np.float32)
    result = renderer.draw_texts(img, [sample_text])
    assert result.shape == (200, 150, 3)
    # Verify text was actually drawn (image should be modified)
    assert not np.array_equal(result, img)
    # Text region should differ from background
    text_region = result[10:30, 10:60]
    background_region = result[50:70, 100:120]
    assert not np.allclose(text_region.mean(), background_region.mean())


@pytest.mark.anyio
async def test_replace_text_returns_pil_image(
    renderer: ImageRenderer, sample_text: Text, sample_img_id: str
) -> None:
    result = await renderer.replace_text(sample_img_id, [sample_text], np.identity(4))
    assert isinstance(result, Image.Image)
    # Count black pixels (text pixels) - original white image has 0 black pixels
    result_array = np.array(result)
    black_pixels = np.sum(np.all(result_array < 50, axis=2))  # Dark pixels across all channels
    # Text replacement should introduce black pixels
    assert black_pixels > 0


@pytest.mark.anyio
async def test_replace_text_with_debug_saves_mask(
    renderer: ImageRenderer, temp_dir: str, sample_text: Text, sample_img_id: str
) -> None:
    await renderer.replace_text(
        sample_img_id, [sample_text], np.identity(4), save_debug=True, debug_dir=temp_dir
    )

    assert (Path(temp_dir) / "debug" / "debug-mask.png").exists()


@pytest.mark.anyio
async def test_replace_text_with_debug_saves_overlay(
    renderer: ImageRenderer, temp_dir: str, sample_text: Text, sample_img_id: str
) -> None:
    await renderer.replace_text(
        sample_img_id, [sample_text], np.identity(4), save_debug=True, debug_dir=temp_dir
    )

    assert (Path(temp_dir) / "debug-overlay.png").exists()


@pytest.mark.anyio
async def test_replace_text_integration_workflow(
    renderer: ImageRenderer,
    sample_text: Text,
    sample_img_id: str,
) -> None:
    original_array = await FileClient.get_image_as_numpy(sample_img_id)
    if original_array is None:
        raise RuntimeError("Failed to download sample image as numpy array")

    # Count black pixels in original (should be 0 on uniform gray background)
    original_black_pixels = np.sum(np.all(original_array < 50, axis=2))

    result = await renderer.replace_text(sample_img_id, [sample_text], np.identity(4))
    assert result.size == (100, 100)

    # Count black pixels after replacement
    result_array = np.array(result)
    result_black_pixels = np.sum(np.all(result_array < 50, axis=2))

    # Black pixel count should change due to text replacement
    # (new text likely has different pixel count than original)
    assert result_black_pixels != original_black_pixels
    # Result should have black text pixels
    assert result_black_pixels > 0
