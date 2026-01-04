from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator


class BBoxPoint(BaseModel):
    """Image-space point used by OCR/document polygons."""

    x: float
    """X coordinate in pixels."""
    y: float
    """Y coordinate in pixels."""


class FontInfo(BaseModel):
    """Font detection results from `traenslenzor.font_detector.mcp.detect_font`."""

    detectedFont: str
    """Font family name."""
    font_size: int
    """Font size in pixels."""


class TranslationInfo(BaseModel):
    """Translation results from `traenslenzor.translator.mcp.translate`."""

    translatedText: str
    """Translated text."""


class OCRBase(BaseModel):
    """Base OCR data present in all text items."""

    extractedText: str
    """Raw text from PaddleOCR `rec_texts`."""
    confidence: float
    """PaddleOCR confidence score from `rec_scores`."""
    bbox: list[BBoxPoint]
    """PaddleOCR polygon from `rec_polys` (4 points ordered UL, UR, LR, LL)."""
    color: tuple[int, int, int] | None = None
    """Reserved for OCR text color (not populated by current tools)."""


class OCRTextItem(OCRBase):
    """Text item directly from OCR - no processing yet.

    Produced by: `traenslenzor.text_extractor.mcp.extract_text`
    Consumed by: Font Detector, Translator
    """

    type: Literal["ocr"] = "ocr"


class FontDetectedItem(OCRBase):
    """Text item with font detection but no translation.

    Produced by: Font Detector (from OCRTextItem)
    Consumed by: Translator (to produce RenderReadyItem)
    """

    type: Literal["font_detected"] = "font_detected"
    font: FontInfo


class TranslatedOnlyItem(OCRBase):
    """Text item with translation but no font detection.

    Produced by: Translator (from OCRTextItem)
    Consumed by: Font Detector (to produce RenderReadyItem)
    """

    type: Literal["translated_only"] = "translated_only"
    translation: TranslationInfo


class RenderReadyItem(OCRBase):
    """Fully processed text item ready for rendering.

    Produced by:
        - Font Detector (from TranslatedOnlyItem)
        - Translator (from FontDetectedItem)
    Consumed by: `traenslenzor.image_renderer.mcp.replace_text`
    """

    type: Literal["render_ready"] = "render_ready"
    font: FontInfo
    translation: TranslationInfo


# Discriminated Union for all possible text item states
TextItem = Annotated[
    Union[OCRTextItem, FontDetectedItem, TranslatedOnlyItem, RenderReadyItem],
    Discriminator("type"),
]

# Type aliases for processor input/output signatures
NeedsFontDetection = OCRTextItem | TranslatedOnlyItem
NeedsTranslation = OCRTextItem | FontDetectedItem
HasFontInfo = FontDetectedItem | RenderReadyItem
HasTranslation = TranslatedOnlyItem | RenderReadyItem


def add_font_info(item: TextItem, font: FontInfo) -> FontDetectedItem | RenderReadyItem:
    """Add font detection results to any text item.

    Args:
        item: Any text item in the pipeline
        font: Font detection results to add

    Returns:
        FontDetectedItem if item had no translation, RenderReadyItem if it did
    """
    base_data = item.model_dump(exclude={"type", "font", "translation"})

    if isinstance(item, (TranslatedOnlyItem, RenderReadyItem)):
        # Already has translation -> becomes RenderReady
        return RenderReadyItem(**base_data, font=font, translation=item.translation)
    else:
        # No translation yet -> becomes FontDetected
        return FontDetectedItem(**base_data, font=font)


def add_translation(
    item: TextItem, translation: TranslationInfo
) -> TranslatedOnlyItem | RenderReadyItem:
    """Add translation results to any text item.

    Args:
        item: Any text item in the pipeline
        translation: Translation results to add

    Returns:
        TranslatedOnlyItem if item had no font info, RenderReadyItem if it did
    """
    base_data = item.model_dump(exclude={"type", "font", "translation"})

    if isinstance(item, (FontDetectedItem, RenderReadyItem)):
        # Already has font -> becomes RenderReady
        return RenderReadyItem(**base_data, font=item.font, translation=translation)
    else:
        # No font yet -> becomes TranslatedOnly
        return TranslatedOnlyItem(**base_data, translation=translation)


def is_render_ready(item: TextItem) -> bool:
    """Check if an item is ready for rendering."""
    return isinstance(item, RenderReadyItem)


def filter_render_ready(items: list[TextItem]) -> list[RenderReadyItem]:
    """Filter a list to only render-ready items."""
    return [item for item in items if isinstance(item, RenderReadyItem)]


def all_render_ready(items: list[TextItem]) -> bool:
    """Check if all items in the list are ready for rendering."""
    return all(is_render_ready(item) for item in items)


class ExtractedDocument(BaseModel):
    """Deskewed document metadata from `traenslenzor.text_extractor.mcp.extract_text`."""

    id: str
    """File id for the deskewed image uploaded in `traenslenzor.text_extractor.mcp.extract_text`."""
    transformation_matrix: list[list[float]]
    """3x3 transformation matrix from OpenCV's getPerspectiveTransform used by `traenslenzor.image_renderer.mcp_server.replace_text` to transform rendered text back to original image space."""
    documentCoordinates: list[BBoxPoint]
    """Document polygon for the deskewed image."""


class SessionState(BaseModel):
    """Session state stored by `traenslenzor.file_server.server` and MCP tools."""

    rawDocumentId: str | None = None
    """Raw document file id set by `traenslenzor.supervisor.tools.document_loader.document_loader`."""
    extractedDocument: ExtractedDocument | None = None
    """Deskewed document metadata set by `traenslenzor.text_extractor.mcp.extract_text`."""
    renderedDocumentId: str | None = None
    """Rendered image id set by `traenslenzor.image_renderer.mcp.replace_text`."""
    text: list[TextItem] | None = None
    """OCR items set by `traenslenzor.text_extractor.mcp.extract_text` and enriched by `traenslenzor.font_detector.mcp.detect_font` and `traenslenzor.translator.mcp.translate`."""
    language: str | None = None
    """Target language set by `traenslenzor.supervisor.tools.set_target_lang.set_target_language`."""
    class_probabilities: dict[str, float] | None = None
    """Class probabilities set by `traenslenzor.doc_class_detector.mcp.classify_document` or `traenslenzor.doc_classifier.mcp_integration.mcp_server.classify_document`."""


def initialize_session() -> SessionState:
    """Create a new empty session for the file server."""
    return SessionState()


class SessionProgressStep(BaseModel):
    """Represents a single workflow step and its completion status."""

    label: str
    done: bool
    detail: str | None = None


ProgressStage = Literal[
    "awaiting_document",
    "detecting_language",
    "extracting_text",
    "translating",
    "detecting_font",
    "classifying",
    "rendering",
]


class SessionProgress(BaseModel):
    """Derived progress summary for a session."""

    session_id: str
    stage: ProgressStage
    completed_steps: int
    total_steps: int
    steps: list[SessionProgressStep]
