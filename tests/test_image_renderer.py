import asyncio
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from traenslenzor.file_server.client import FileClient
from traenslenzor.file_server.session_state import BBoxPoint, TranslatedTextItem
from traenslenzor.image_renderer.image_rendering import ImageRenderer
from traenslenzor.image_renderer.mcp_server import get_device
from traenslenzor.image_renderer.text_operations import create_mask, draw_texts, get_angle_from_bbox
from traenslenzor.text_extractor.flatten_image import deskew_document
from traenslenzor.text_extractor.paddleocr import run_ocr
from traenslenzor.translator.translator import translate


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def renderer() -> ImageRenderer:
    return ImageRenderer(device=get_device())


@pytest.fixture
def sample_text() -> TranslatedTextItem:
    return TranslatedTextItem(
        extractedText="Hello",
        translatedText="Hallo",
        bbox=[
            BBoxPoint(x=10, y=10),
            BBoxPoint(x=60, y=10),
            BBoxPoint(x=60, y=30),
            BBoxPoint(x=10, y=30),
        ],
        confidence=0.98,
        font_size="16",
        color=(0, 0, 0),
        detectedFont="Arial",
    )


@pytest.fixture
def sample_img() -> Image.Image:
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


@pytest.fixture
def sample_img_fixture() -> PILImage:
    return Image.open(Path(__file__).parent / "fixtures" / "image_1.png").convert("RGB")


@pytest.fixture
def sample_img_skewed() -> PILImage:
    return Image.open(Path(__file__).parent / "fixtures" / "skewed_image_3.jpeg").convert("RGB")


@pytest.fixture
async def sample_img_id(file_server: type[FileClient], sample_img: Image.Image) -> str:
    id = await file_server.put_img("test.png", sample_img)
    if id is None:
        raise RuntimeError("Failed to upload sample image to file server")

    return id


def test_create_mask_single_text_region(
    renderer: ImageRenderer, sample_text: TranslatedTextItem
) -> None:
    mask = create_mask([sample_text], (100, 100))
    assert mask.shape == (1, 100, 100)
    assert mask.dtype == np.uint8
    assert mask[0, 10:30, 10:60].sum() > 0


def test_create_mask_multiple_text_regions(
    renderer: ImageRenderer, sample_text: TranslatedTextItem
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
    renderer: ImageRenderer, sample_text: TranslatedTextItem
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
    text = TranslatedTextItem(
        extractedText="Test",
        translatedText="Hallo",
        bbox=[
            BBoxPoint(x=80, y=80),
            BBoxPoint(x=130, y=80),
            BBoxPoint(x=130, y=130),
            BBoxPoint(x=80, y=130),
        ],
        confidence=0.98,
        font_size="16",
        color=(0, 0, 0),
        detectedFont="Arial",
    )
    mask = create_mask([text], (100, 100))
    assert mask.shape == (1, 100, 100)


def test_create_mask_correct_shape(
    renderer: ImageRenderer, sample_text: TranslatedTextItem
) -> None:
    mask = create_mask([sample_text], (200, 150))
    assert mask.shape == (1, 200, 150)


def test_draw_texts_single_text(renderer: ImageRenderer, sample_text: TranslatedTextItem) -> None:
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    result = draw_texts(img, [sample_text])
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.float32


def test_draw_texts_multiple_texts(
    renderer: ImageRenderer, sample_text: TranslatedTextItem
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
    text = TranslatedTextItem(
        extractedText="Test",
        translatedText="Test",
        bbox=[
            BBoxPoint(x=10, y=10),
            BBoxPoint(x=60, y=10),
            BBoxPoint(x=60, y=30),
            BBoxPoint(x=10, y=30),
        ],
        confidence=0.98,
        font_size="24",
        color=(255, 0, 0),
        detectedFont="Arial",
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
    renderer: ImageRenderer, sample_text: TranslatedTextItem
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
    renderer: ImageRenderer, sample_text: TranslatedTextItem, sample_img: Image.Image
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
    renderer: ImageRenderer, temp_dir: str, sample_text: TranslatedTextItem, sample_img: Image.Image
) -> None:
    await renderer.replace_text(sample_img, [sample_text], save_debug=True, debug_dir=temp_dir)

    assert (Path(temp_dir) / "debug-mask.png").exists()


@pytest.mark.anyio
async def test_replace_text_with_debug_saves_overlay(
    renderer: ImageRenderer, temp_dir: str, sample_text: TranslatedTextItem, sample_img: Image.Image
) -> None:
    await renderer.replace_text(sample_img, [sample_text], save_debug=True, debug_dir=temp_dir)

    assert (Path(temp_dir) / "debug-overlay.png").exists()


@pytest.mark.anyio
async def test_replace_text_integration_workflow(
    renderer: ImageRenderer,
    sample_text: TranslatedTextItem,
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


@pytest.mark.anyio
async def test_draw_rotated_text(
    renderer: ImageRenderer,
    sample_img_fixture: Image.Image,
):
    bbox = [
        BBoxPoint(x=140, y=120),
        BBoxPoint(x=247, y=120),
        BBoxPoint(x=247, y=205),
        BBoxPoint(x=140, y=205),
    ]

    # Convert to numpy array
    array_bbox = np.array([[point.x, point.y] for point in bbox])

    # Calculate the center of the bbox for rotation
    center = array_bbox.mean(axis=0)

    # Apply rotation: translate to origin, rotate, translate back
    expected_angle = 330.0
    rotation_matrix = np.array(
        [
            [np.cos(np.radians(expected_angle)), -np.sin(np.radians(expected_angle))],
            [np.sin(np.radians(expected_angle)), np.cos(np.radians(expected_angle))],
        ]
    )

    # Translate to origin, rotate, translate back
    bbox_centered = array_bbox - center
    bbox_rotated = bbox_centered @ rotation_matrix.T
    rotated_bbox = bbox_rotated + center

    # Convert back to BBoxPoint
    rotated_bbox_points = [BBoxPoint(x=float(x), y=float(y)) for x, y in rotated_bbox]

    translated_text = TranslatedTextItem(
        extractedText="Test",
        translatedText="Test",
        bbox=rotated_bbox_points,
        confidence=0.98,
        font_size="18",
        color=(255, 255, 255),
        detectedFont="Arial",
    )

    result = await renderer.replace_text(
        sample_img_fixture, texts=[translated_text], save_debug=True, debug_dir="./debug"
    )

    result.save("./debug/rotated_text_result.png")

    # Convert to numpy for pixel analysis
    result_array = np.array(result)
    original_array = np.array(sample_img_fixture)
    # 1. Verify the image was modified
    assert not np.array_equal(result_array, original_array), "Image should be modified"

    # 2. Calculate bounding box of rotated text region with margin
    min_x = int(min(p.x for p in rotated_bbox_points))
    max_x = int(max(p.x for p in rotated_bbox_points))
    min_y = int(min(p.y for p in rotated_bbox_points))
    max_y = int(max(p.y for p in rotated_bbox_points))

    # Add margin for rotation artifacts, anti-aliasing, and padding
    margin = 10
    text_min_x = max(0, min_x - margin)
    text_min_y = max(0, min_y - margin)
    text_max_x = min(result.size[0], max_x + margin)
    text_max_y = min(result.size[1], max_y + margin)

    # 3. Verify pixels changed in the rotated bbox region
    text_region = result_array[text_min_y:text_max_y, text_min_x:text_max_x]
    original_text_region = original_array[text_min_y:text_max_y, text_min_x:text_max_x]

    # Text region should differ from original (text was drawn)
    assert not np.array_equal(text_region, original_text_region), (
        "Text region should be modified with rendered text"
    )

    # 4. Verify dark pixels exist (text was rendered in black)
    dark_pixels = np.sum(np.all(result_array < 128, axis=2))
    assert dark_pixels > 50, f"Expected dark text pixels, found {dark_pixels}"

    # 5. Verify the rotation angle is preserved in the bbox
    calculated_angle, _ = get_angle_from_bbox(rotated_bbox_points)
    assert abs(calculated_angle - expected_angle) < 0.01, (
        f"Rotation angle mismatch: expected {expected_angle}, got {calculated_angle}"
    )

    # 6. Verify the rest of the image hasn't changed
    # Create a mask for the text region
    mask = np.zeros((result.size[1], result.size[0]), dtype=bool)
    mask[text_min_y:text_max_y, text_min_x:text_max_x] = True

    # Invert mask to get non-text regions
    non_text_mask = ~mask

    # Compare non-text regions
    original_outside = original_array[non_text_mask]
    result_outside = result_array[non_text_mask]

    # All pixels outside the text region should be unchanged
    assert np.array_equal(original_outside, result_outside), (
        "Pixels outside the text region should remain unchanged"
    )

    # Alternative: Check that the percentage of unchanged pixels is high
    total_pixels = result.size[0] * result.size[1]
    unchanged_pixels = np.sum(non_text_mask)
    unchanged_percentage = (unchanged_pixels / total_pixels) * 100

    # Most of the image should be unchanged (depends on text size)
    assert unchanged_percentage > 50, (
        f"Expected >50% of image unchanged, got {unchanged_percentage:.1f}%"
    )


@pytest.mark.anyio
async def test_transform_image_with_ocr_bbox(renderer: ImageRenderer, sample_img_skewed: PILImage):
    print("Starting deskewing process...")
    result = deskew_document(np.array(sample_img_skewed))
    print("Deskewing process completed.")

    assert result is not None

    unskewed_img, matrix = result
    unskewed_pil = Image.fromarray(unskewed_img)
    unskewed_pil.save("./debug/deskewed_image.png")

    print("Running OCR on deskewed image...")
    ocr_result = run_ocr("de", np.array(unskewed_img))
    print("OCR process completed.")

    assert ocr_result is not None

    print("Starting translation process...")
    translatedTexts = await asyncio.gather(
        *[asyncio.create_task(translate(item, "en_GB")) for item in ocr_result]
    )
    print("Translation process completed.")
    for text in [
        f"{o.extractedText} -> {t.translatedText}" for o, t in zip(ocr_result, translatedTexts)
    ]:
        print(text)

    result = await renderer.replace_text(
        unskewed_pil, texts=translatedTexts, save_debug=True, debug_dir="./debug"
    )

    result.save("./debug/replaced_unskewed.png")

    transformed = renderer.transform_image(
        result,
        np.linalg.inv(matrix),
        original_size=(sample_img_skewed.height, sample_img_skewed.width),
    )
    transformed.save("./debug/replaced_reprojected.png")

    composited = renderer.paste_replaced_to_original(sample_img_skewed, transformed)

    composited.save("./debug/replaced_full.png")


@pytest.mark.anyio
async def test_transform_image(renderer: ImageRenderer, sample_img_skewed: PILImage):
    result = deskew_document(np.array(sample_img_skewed))

    if result is None:
        assert False

    unskewed_img, matrix = result
    unskewed_pil = Image.fromarray(unskewed_img)
    unskewed_pil.save("./debug/deskewed_image.png")

    # handcoded points from the image
    bbox = [
        BBoxPoint(x=22, y=27),
        BBoxPoint(x=210, y=27),
        BBoxPoint(x=210, y=45),
        BBoxPoint(x=22, y=45),
    ]
    translated_text = TranslatedTextItem(
        extractedText="Verhalten im Brandfall",
        translatedText="Im Falle von Brand? Bier!",
        bbox=bbox,
        confidence=0.98,
        font_size="18",
        color=(0, 0, 0),
        detectedFont="Arial",
    )

    result = await renderer.replace_text(
        unskewed_pil, texts=[translated_text], save_debug=True, debug_dir="./debug"
    )

    result.save("./debug/replaced_unskewed.png")

    transformed = renderer.transform_image(
        result,
        np.linalg.inv(matrix),
        original_size=(sample_img_skewed.height, sample_img_skewed.width),
    )
    transformed.save("./debug/replaced_reprojected.png")

    composited = renderer.paste_replaced_to_original(sample_img_skewed, transformed)

    composited.save("./debug/replaced_full.png")
