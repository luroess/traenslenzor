import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from traenslenzor.file_server.session_state import (
    BBoxPoint,
    FontInfo,
    RenderReadyItem,
    TranslationInfo,
    add_font_info,
)
from traenslenzor.image_renderer.image_rendering import ImageRenderer
from traenslenzor.image_renderer.mcp_server import get_device
from traenslenzor.image_renderer.text_operations import get_angle_from_bbox
from traenslenzor.text_extractor.flatten_image import deskew_document
from traenslenzor.text_extractor.ocr import run_ocr
from traenslenzor.translator.translator import translate_all

# Skip tests that require model downloads when running in CI
IN_CI = os.getenv("CI") == "true"


@pytest.fixture
def renderer() -> ImageRenderer:
    return ImageRenderer(device=get_device())


@pytest.fixture
def sample_img_fixture() -> PILImage:
    return Image.open(Path(__file__).parent / "fixtures" / "image_1.png").convert("RGB")


@pytest.fixture
def sample_img_skewed() -> PILImage:
    return Image.open(Path(__file__).parent / "fixtures" / "skewed_image_3.jpeg").convert("RGB")


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
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

    translated_text = RenderReadyItem(
        extractedText="Test",
        bbox=rotated_bbox_points,
        confidence=0.98,
        color=(255, 255, 255),
        font=FontInfo(detectedFont="Arial", font_size=18),
        translation=TranslationInfo(translatedText="Test"),
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


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_transform_image_with_ocr_bbox(renderer: ImageRenderer, sample_img_skewed: PILImage):
    print("Starting deskewing process...")
    result = deskew_document(np.array(sample_img_skewed))
    print("Deskewing process completed.")

    assert result is not None

    unskewed_img, matrix, pts = result
    unskewed_pil = Image.fromarray(unskewed_img)
    unskewed_pil.save("./debug/deskewed_image.png")

    print("Running OCR on deskewed image...")
    ocr_result = run_ocr("de", np.array(unskewed_img))
    print("OCR process completed.")

    assert ocr_result is not None

    print("Starting translation process...")
    translatedTexts = await translate_all(ocr_result, "en_GB")
    print("Translation process completed.")
    for text in [
        f"{o.extractedText} -> {t.translation.translatedText}"
        for o, t in zip(ocr_result, translatedTexts)
    ]:
        print(text)

    # Add font info to make texts render-ready
    render_ready_texts = []
    for t in translatedTexts:
        if t.type == "render_ready":
            render_ready_texts.append(t)
        else:
            # Add default font info to translated-only items
            font_info = FontInfo(detectedFont="Arial", font_size=16)
            render_ready_texts.append(add_font_info(t, font_info))

    result = await renderer.replace_text(
        unskewed_pil, texts=render_ready_texts, save_debug=True, debug_dir="./debug"
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


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_transform_image(renderer: ImageRenderer, sample_img_skewed: PILImage):
    result = deskew_document(np.array(sample_img_skewed))

    if result is None:
        assert False

    unskewed_img, matrix, pts = result
    unskewed_pil = Image.fromarray(unskewed_img)
    unskewed_pil.save("./debug/deskewed_image.png")

    # handcoded points from the image
    bbox = [
        BBoxPoint(x=22, y=27),
        BBoxPoint(x=210, y=27),
        BBoxPoint(x=210, y=45),
        BBoxPoint(x=22, y=45),
    ]
    translated_text = RenderReadyItem(
        extractedText="Verhalten im Brandfall",
        bbox=bbox,
        confidence=0.98,
        color=(0, 0, 0),
        font=FontInfo(detectedFont="Arial", font_size=18),
        translation=TranslationInfo(translatedText="Im Falle von Brand? Bier!"),
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
