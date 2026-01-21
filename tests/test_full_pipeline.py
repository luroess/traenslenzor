import os

import numpy as np
import pytest
from fastmcp import Client
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import ExtractedDocument, SessionState
from traenslenzor.font_detector.mcp import font_detector
from traenslenzor.image_renderer.mcp_server import image_renderer_mcp
from traenslenzor.text_extractor.flatten_image import deskew_document
from traenslenzor.text_extractor.mcp import text_extractor
from traenslenzor.translator.mcp import translator

# Skip tests that require model downloads when running in CI
IN_CI = os.getenv("CI") == "true"


def create_test_image(texts: list[str]) -> PILImage:
    """
    Create a realistic test image with rotated white document on noisy background.

    Returns:
        tuple: (original_skewed_image, deskewed_image, text_item_definitions)
    """
    # Phase 1: Define document structure
    doc_width, doc_height = 800, 600
    bg_width, bg_height = 1200, 900
    rotation_angle = 10  # degrees

    # Phase 2: Create base white document with text at exact bbox positions
    document = Image.new("RGBA", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(document)

    # Load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()

    num_texts = len(texts)
    # Draw text at exact bbox positions (top-left aligned)
    for i, text in enumerate(texts):
        y = (doc_height / num_texts) * i + 10
        # Get top-left corner of bbox
        draw.text((10, y), text, fill=(0, 0, 0), font=font)

    print(f"Created base document: {doc_width}x{doc_height}")

    # Phase 3: Rotate the document and paste on noisy background
    # Rotate with expand=True to avoid cropping
    rotated_doc = document.rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255, 0))

    # Create noisy background
    noise = np.random.randint(0, 100, (bg_height, bg_width, 3), dtype=np.uint8)
    background = Image.fromarray(noise)

    # Create canvas and paste background
    canvas = Image.new("RGBA", (bg_width, bg_height))
    canvas.paste(background, (0, 0))

    # Position the rotated document in the center
    paste_x = (bg_width - rotated_doc.width) // 2
    paste_y = (bg_height - rotated_doc.height) // 2
    canvas.paste(rotated_doc, (paste_x, paste_y), rotated_doc)

    print(f"Rotated document by {rotation_angle}° and pasted at ({paste_x}, {paste_y})")

    return canvas


async def create_extracted_document(
    img: PILImage,
) -> ExtractedDocument:
    deskew_result = deskew_document(np.array(img))
    assert deskew_result is not None, "Failed to deskew document - deskew_document returned None"

    deskewed_img, transformationMatrix = deskew_result

    id = await FileClient.put_img("deskewed_test_image.png", Image.fromarray(deskewed_img))
    assert id is not None, "Failed to upload deskewed image to file server"

    document = ExtractedDocument(
        id=id,
        transformation_matrix=transformationMatrix.tolist(),
    )

    return document


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_text_extractor_tool(file_server):
    """Test the text_extractor MCP tool independently."""
    test_texts = [
        "Hello world, how are you doing today?",
        "This is a test document for translation.",
        "Machine translation works great!",
    ]

    test_img = create_test_image(test_texts)

    raw_doc_id = await FileClient.put_img("test_image.png", test_img)
    assert raw_doc_id is not None, "Failed to upload test image to file server"

    # Create session with the raw document ID
    session = SessionState(rawDocumentId=raw_doc_id)
    session_id = await SessionClient.create(session)

    # Test text extraction
    async with Client(text_extractor) as text_extractor_client:
        result = await text_extractor_client.call_tool(
            "extract_text",
            {"session_id": session_id},
        )
        assert result is not None, "Text extraction returned None"

    session_after_extraction = await SessionClient.get(session_id)

    assert session_after_extraction.extractedDocument is not None, (
        "No extracted document found after text extraction"
    )
    assert session_after_extraction.text is not None, "No text items found after extraction"
    assert len(session_after_extraction.text) > 0, "Text items list is empty"

    for text in session_after_extraction.text:
        assert text.type == "ocr", f"Expected text type 'ocr', got '{text.type}'"
        assert len(text.extractedText) > 0, "Extracted text is empty"
        assert text.bbox is not None, "Bounding box is None"
        assert len(text.bbox) == 4, f"Expected 4 bbox points, got {len(text.bbox)}"


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_font_detector_tool(file_server):
    """Test the font_detector MCP tool independently."""
    test_texts = [
        "Hello world, how are you doing today?",
        "This is a test document for translation.",
        "Machine translation works great!",
    ]

    test_img = create_test_image(test_texts)

    raw_doc_id = await FileClient.put_img("test_image_font.png", test_img)
    assert raw_doc_id is not None, "Failed to upload test image to file server"

    # Create session with the raw document ID
    session = SessionState(rawDocumentId=raw_doc_id)
    session_id = await SessionClient.create(session)

    # First, run text extraction (dependency)
    async with Client(text_extractor) as text_extractor_client:
        await text_extractor_client.call_tool(
            "extract_text",
            {"session_id": session_id},
        )

    # Test font detection
    async with Client(font_detector) as font_detector_client:
        result = await font_detector_client.call_tool(
            "detect_font",
            {"session_id": session_id},
        )
        assert result is not None, "Font detection returned None"

    session_after_detection = await SessionClient.get(session_id)
    assert session_after_detection.text is not None, "No text items found after font detection"
    assert len(session_after_detection.text) > 0, "Text items list is empty after font detection"

    for text in session_after_detection.text:
        assert text.type == "font_detected", (
            f"Expected text type 'font_detected', got '{text.type}'"
        )
        assert len(text.extractedText) > 0, "Extracted text is empty after font detection"
        assert text.font is not None, "Font info is None"
        assert text.font.font_size is not None, "Font size was not detected"
        assert text.font.font_size > 0, f"Font size should be positive, got {text.font.font_size}"
        assert text.font.detectedFont is not None, "Font was not detected"
        assert len(text.font.detectedFont) > 0, "Detected font name is empty"


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_translator_tool(file_server):
    """Test the translator MCP tool independently."""
    test_texts = [
        "Hello world, how are you doing today?",
        "This is a test document for translation.",
        "Machine translation works great!",
    ]

    test_img = create_test_image(test_texts)

    raw_doc_id = await FileClient.put_img("test_image_translate.png", test_img)
    assert raw_doc_id is not None, "Failed to upload test image to file server"

    # Create session with the raw document ID
    session = SessionState(rawDocumentId=raw_doc_id)
    session_id = await SessionClient.create(session)

    # First, run text extraction (dependency)
    async with Client(text_extractor) as text_extractor_client:
        await text_extractor_client.call_tool(
            "extract_text",
            {"session_id": session_id},
        )

    # Test translation
    async with Client(translator) as translator_client:
        result = await translator_client.call_tool(
            "translate",
            {"session_id": session_id},
        )
        assert result is not None, "Translation returned None"

    session_after_translation = await SessionClient.get(session_id)
    assert session_after_translation.text is not None, "No text items found after translation"
    assert len(session_after_translation.text) > 0, "Text items list is empty after translation"

    for text in session_after_translation.text:
        # Translation can produce either 'translated_only' or 'render_ready' depending on font info
        assert text.type in ["translated_only", "render_ready"], (
            f"Expected text type 'translated_only' or 'render_ready', got '{text.type}'"
        )
        assert len(text.extractedText) > 0, "Extracted text is empty after translation"
        assert text.translation is not None, "Translation info is None"
        assert text.translation.translatedText is not None, "Translated text is None"
        assert len(text.translation.translatedText) > 0, "Translated text is empty"


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_image_renderer_tool(file_server):
    """Test the image_renderer MCP tool independently."""
    test_texts = [
        "Hello world, how are you doing today?",
        "This is a test document for translation.",
        "Machine translation works great!",
    ]

    test_img = create_test_image(test_texts)

    raw_doc_id = await FileClient.put_img("test_image_render.png", test_img)
    assert raw_doc_id is not None, "Failed to upload test image to file server"

    # Create session with the raw document ID
    session = SessionState(rawDocumentId=raw_doc_id)
    session_id = await SessionClient.create(session)

    # First, run text extraction (dependency 1)
    async with Client(text_extractor) as text_extractor_client:
        await text_extractor_client.call_tool(
            "extract_text",
            {"session_id": session_id},
        )

    # Second, run font detection (dependency 2)
    async with Client(font_detector) as font_detector_client:
        await font_detector_client.call_tool(
            "detect_font",
            {"session_id": session_id},
        )

    # Third, run translation (dependency 3)
    async with Client(translator) as translator_client:
        await translator_client.call_tool(
            "translate",
            {"session_id": session_id},
        )

    # Verify we have render_ready items
    session_before_rendering = await SessionClient.get(session_id)
    assert session_before_rendering.text is not None, "No text items before rendering"
    for text in session_before_rendering.text:
        assert text.type == "render_ready", (
            f"Expected text type 'render_ready' before rendering, got '{text.type}'"
        )

    # Test image rendering
    async with Client(image_renderer_mcp) as image_renderer_client:
        result = await image_renderer_client.call_tool(
            "replace_text",
            {"session_id": session_id},
        )
        assert result is not None, "Image rendering returned None"

    session_after_rendering = await SessionClient.get(session_id)
    assert session_after_rendering.renderedDocumentId is not None, (
        "No rendered document ID found after rendering"
    )

    # Verify the rendered image exists and can be retrieved
    rendered_image = await FileClient.get_image(session_after_rendering.renderedDocumentId)
    assert rendered_image is not None, "Failed to retrieve rendered image"
    assert rendered_image.width > 0, "Rendered image has invalid width"
    assert rendered_image.height > 0, "Rendered image has invalid height"


@pytest.mark.skipif(IN_CI, reason="Skip model download in CI")
@pytest.mark.anyio
async def test_full_pipeline(file_server):
    """Test the full pipeline with all MCP tools in sequence."""
    test_texts = [
        "Hello world, how are you doing today?",
        "This is a test document for translation.",
        "Machine translation works great!",
    ]

    test_img = create_test_image(test_texts)

    raw_doc_id = await FileClient.put_img("test_image.png", test_img)
    assert raw_doc_id is not None, "Failed to upload test image to file server"

    # Create session with the raw document ID
    session = SessionState(rawDocumentId=raw_doc_id)
    session_id = await SessionClient.create(session)

    async with Client(text_extractor) as text_extractor_client:
        await text_extractor_client.call_tool(
            "extract_text",
            {"session_id": session_id},
        )

    session_after_extraction = await SessionClient.get(session_id)

    assert session_after_extraction.extractedDocument is not None, (
        "No extracted document found after text extraction"
    )
    assert session_after_extraction.text is not None, "No text items found after extraction"

    for text in session_after_extraction.text:
        assert text.type == "ocr", f"Expected text type 'ocr', got '{text.type}'"
        assert len(text.extractedText) > 0, "Extracted text is empty"

    async with Client(font_detector) as font_detector_client:
        await font_detector_client.call_tool(
            "detect_font",
            {"session_id": session_id},
        )

    session_after_detection = await SessionClient.get(session_id)
    assert session_after_detection.text is not None, "No text items found after font detection"
    for text in session_after_detection.text:
        assert text.type == "font_detected", (
            f"Expected text type 'font_detected', got '{text.type}'"
        )
        assert len(text.extractedText) > 0, "Extracted text is empty after font detection"
        assert text.font.font_size is not None, "Font size was not detected"
        assert text.font.detectedFont is not None, "Font was not detected"

    async with Client(translator) as translator_client:
        await translator_client.call_tool(
            "translate",
            {"session_id": session_id},
        )

    session_after_translation = await SessionClient.get(session_id)
    assert session_after_translation.text is not None, "No text items found after translation"
    for text in session_after_translation.text:
        assert text.type == "render_ready", f"Expected text type 'render_ready', got '{text.type}'"
        assert text.font.font_size is not None, "Font size missing after translation"
        assert text.font.detectedFont is not None, "Detected font missing after translation"
        assert text.translation.translatedText is not None, "Translated text is None"

    async with Client(image_renderer_mcp) as image_renderer_client:
        await image_renderer_client.call_tool(
            "replace_text",
            {"session_id": session_id},
        )

    session_after_rendering = await SessionClient.get(session_id)
    assert session_after_rendering.renderedDocumentId is not None, (
        "No rendered document ID found after rendering"
    )
