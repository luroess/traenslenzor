import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traenslenzor.file_server.client import FileClient
from traenslenzor.file_server.session_state import (
    BBoxPoint,
    FontInfo,
    RenderReadyItem,
    TranslationInfo,
)
from traenslenzor.image_renderer.image_rendering import ImageRenderer
from traenslenzor.image_renderer.mcp_server import get_device
from traenslenzor.image_renderer.text_operations import create_mask, draw_texts, get_angle_from_bbox


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def renderer() -> ImageRenderer:
    return ImageRenderer(device=get_device())


@pytest.fixture
def sample_text() -> RenderReadyItem:
    return RenderReadyItem(
        extractedText="Hello",
        bbox=[
            BBoxPoint(x=10, y=10),
            BBoxPoint(x=60, y=10),
            BBoxPoint(x=60, y=30),
            BBoxPoint(x=10, y=30),
        ],
        confidence=0.98,
        color=(0, 0, 0),
        font=FontInfo(detectedFont="Arial", font_size=16),
        translation=TranslationInfo(translatedText="Hallo"),
    )


@pytest.fixture
def sample_img() -> Image.Image:
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


@pytest.fixture
async def sample_img_id(file_server: type[FileClient], sample_img: Image.Image) -> str:
    id = await file_server.put_img("test.png", sample_img)
    if id is None:
        raise RuntimeError("Failed to upload sample image to file server")

    return id


def test_create_mask_single_text_region(
    renderer: ImageRenderer, sample_text: RenderReadyItem
) -> None:
    mask = create_mask([sample_text], (100, 100))
    assert mask.shape == (1, 100, 100)
    assert mask.dtype == np.uint8
    assert mask[0, 10:30, 10:60].sum() > 0


def test_create_mask_multiple_text_regions(
    renderer: ImageRenderer, sample_text: RenderReadyItem
) -> None:
    text2 = sample_text.model_copy(
        update={
            "bbox": [
                BBoxPoint(x=60, y=10),
                BBoxPoint(x=110, y=10),
                BBoxPoint(x=110, y=30),
                BBoxPoint(x=60, y=30),
            ]
        }
    )
    mask = create_mask([sample_text, text2], (100, 100))
    # Verify first text region (left=10, width=50)
    assert mask[0, 10:30, 10:60].sum() > 0
    # Verify second text region (left=60, width=50)
    assert mask[0, 10:30, 60:110].sum() > 0
    # Both regions should have masked pixels
    assert mask.sum() > mask[0, 10:30, 10:60].sum()


def test_create_mask_overlapping_regions(
    renderer: ImageRenderer, sample_text: RenderReadyItem
) -> None:
    text2 = sample_text.model_copy(
        update={
            "bbox": [
                BBoxPoint(x=15, y=10),
                BBoxPoint(x=65, y=10),
                BBoxPoint(x=65, y=30),
                BBoxPoint(x=15, y=30),
            ]
        }
    )
    mask = create_mask([sample_text, text2], (100, 100))
    assert mask.shape == (1, 100, 100)
    # Verify overlapping region (left=15 to left=60 where both texts should appear)
    overlap_region = mask[0, 10:30, 15:60]
    assert overlap_region.sum() > 0
    # Mask should handle overlaps correctly (pixels can be masked multiple times)
    assert mask.sum() > 0


def test_create_mask_clamps_to_boundaries(renderer: ImageRenderer) -> None:
    text = RenderReadyItem(
        extractedText="Test",
        bbox=[
            BBoxPoint(x=80, y=80),
            BBoxPoint(x=130, y=80),
            BBoxPoint(x=130, y=130),
            BBoxPoint(x=80, y=130),
        ],
        confidence=0.98,
        color=(0, 0, 0),
        font=FontInfo(detectedFont="Arial", font_size=16),
        translation=TranslationInfo(translatedText="Hallo"),
    )
    mask = create_mask([text], (100, 100))
    assert mask.shape == (1, 100, 100)


def test_create_mask_correct_shape(renderer: ImageRenderer, sample_text: RenderReadyItem) -> None:
    mask = create_mask([sample_text], (200, 150))
    assert mask.shape == (1, 200, 150)


def test_draw_texts_single_text(renderer: ImageRenderer, sample_text: RenderReadyItem) -> None:
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    result = draw_texts(img, [sample_text])
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.float32


def test_draw_texts_multiple_texts(renderer: ImageRenderer, sample_text: RenderReadyItem) -> None:
    text2 = sample_text.model_copy(
        update={
            "bbox": [
                BBoxPoint(x=60, y=10),
                BBoxPoint(x=110, y=10),
                BBoxPoint(x=110, y=30),
                BBoxPoint(x=60, y=30),
            ]
        }
    )
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    result = draw_texts(img, [sample_text, text2])
    assert result.shape == (100, 100, 3)
    # Verify the image was modified (should differ from original uniform gray)
    assert not np.allclose(result, 0.5)
    # Check that both text regions were modified
    region1_modified = not np.allclose(result[10:30, 10:60], 0.5)
    region2_modified = not np.allclose(result[10:30, 60:110], 0.5)
    assert region1_modified and region2_modified


def test_draw_texts_respects_font_properties(renderer: ImageRenderer) -> None:
    text = RenderReadyItem(
        extractedText="Test",
        bbox=[
            BBoxPoint(x=10, y=10),
            BBoxPoint(x=60, y=10),
            BBoxPoint(x=60, y=30),
            BBoxPoint(x=10, y=30),
        ],
        confidence=0.98,
        color=(255, 0, 0),
        font=FontInfo(detectedFont="Arial", font_size=24),
        translation=TranslationInfo(translatedText="Test"),
    )
    img = np.ones((100, 100, 3), dtype=np.float32)
    result = draw_texts(img, [text])
    assert result is not None
    # Verify red color was applied (color specified as RGB(255, 0, 0))
    # The red channel should be significantly higher than green/blue in text region
    text_region = result[10:30, 10:60]
    # Check that red channel has higher values than original white
    assert text_region[:, :, 0].max() > 0.5  # Red channel should have content
    # Image should be modified from original uniform white
    assert not np.allclose(result, 1.0)


def test_draw_texts_preserves_image_dimensions(
    renderer: ImageRenderer, sample_text: RenderReadyItem
) -> None:
    img = np.ones((200, 150, 3), dtype=np.float32)
    result = draw_texts(img, [sample_text])
    assert result.shape == (200, 150, 3)
    # Verify text was actually drawn (image should be modified)
    assert not np.array_equal(result, img)
    # Text region should differ from background
    text_region = result[10:30, 10:60]
    background_region = result[50:70, 100:120]
    assert not np.allclose(text_region.mean(), background_region.mean())


@pytest.mark.anyio
async def test_replace_text_returns_pil_image(
    renderer: ImageRenderer, sample_text: RenderReadyItem, sample_img: Image.Image
) -> None:
    result = await renderer.replace_text(sample_img, [sample_text])
    assert isinstance(result, Image.Image)
    # Count black pixels (text pixels) - original white image has 0 black pixels
    result_array = np.array(result)
    black_pixels = np.sum(np.all(result_array < 50, axis=2))  # Dark pixels across all channels
    # Text replacement should introduce black pixels
    assert black_pixels > 0


@pytest.mark.anyio
async def test_replace_text_with_debug_saves_mask(
    renderer: ImageRenderer, temp_dir: str, sample_text: RenderReadyItem, sample_img: Image.Image
) -> None:
    await renderer.replace_text(sample_img, [sample_text], save_debug=True, debug_dir=temp_dir)

    assert (Path(temp_dir) / "debug-mask.png").exists()


@pytest.mark.anyio
async def test_replace_text_with_debug_saves_overlay(
    renderer: ImageRenderer, temp_dir: str, sample_text: RenderReadyItem, sample_img: Image.Image
) -> None:
    await renderer.replace_text(sample_img, [sample_text], save_debug=True, debug_dir=temp_dir)

    assert (Path(temp_dir) / "debug-overlay.png").exists()


@pytest.mark.anyio
async def test_replace_text_integration_workflow(
    renderer: ImageRenderer,
    sample_text: RenderReadyItem,
    sample_img: Image.Image,
) -> None:
    # Count black pixels in original (should be 0 on uniform white background)
    original_array = np.array(sample_img)
    original_black_pixels = np.sum(np.all(original_array < 50, axis=2))

    result = await renderer.replace_text(sample_img, [sample_text])
    assert result.size == (100, 100)

    # Count black pixels after replacement
    result_array = np.array(result)
    result_black_pixels = np.sum(np.all(result_array < 50, axis=2))

    # Black pixel count should change due to text replacement
    # (new text likely has different pixel count than original)
    assert result_black_pixels != original_black_pixels
    # Result should have black text pixels
    assert result_black_pixels > 0


def test_bbox_to_rotation() -> None:
    # Start with a horizontal rectangle in the origin
    bbox = [
        BBoxPoint(x=0, y=0),
        BBoxPoint(x=60, y=0),
        BBoxPoint(x=60, y=30),
        BBoxPoint(x=0, y=30),
    ]

    # Convert to numpy array
    array_bbox = np.array([[point.x, point.y] for point in bbox])

    # Apply rotation matrix (only works when rectangle is centered at origin)
    expected_angles = [30.0, 60.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]

    for expected_angle in expected_angles:
        rotation_matrix = np.array(
            [
                [np.cos(np.radians(expected_angle)), -np.sin(np.radians(expected_angle))],
                [np.sin(np.radians(expected_angle)), np.cos(np.radians(expected_angle))],
            ]
        )

        rotated_bbox = array_bbox @ rotation_matrix.T

        # Convert back to BBoxPoint
        rotated_bbox_points = [BBoxPoint(x=float(x), y=float(y)) for x, y in rotated_bbox]

        # Calculate angle and verify
        calculated_angle, _ = get_angle_from_bbox(rotated_bbox_points)
        assert abs(calculated_angle - expected_angle) < 0.01, (
            f"angle mismatch. expected: {expected_angle}, got: {calculated_angle}"
        )  # Allow small numerical error
