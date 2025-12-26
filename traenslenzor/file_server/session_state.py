from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator


class BBoxPoint(BaseModel):
    """Image-space point used by `traenslenzor.text_extractor.paddleocr.parse_result` and `traenslenzor.text_extractor.mcp.extract_text`."""

    x: float
    """X coordinate in pixels for OCR/document polygons."""
    y: float
    """Y coordinate in pixels for OCR/document polygons."""


class OCRTextItem(BaseModel):
    """OCR text item from `traenslenzor.text_extractor.paddleocr.parse_result`, stored by `traenslenzor.text_extractor.mcp.extract_text`."""

    type: Literal["untranslated"] = "untranslated"
    extractedText: str
    """Raw text from PaddleOCR `rec_texts`."""
    confidence: float
    """PaddleOCR confidence score from `rec_scores`."""
    bbox: list[BBoxPoint]
    """PaddleOCR polygon from `rec_polys` (4 points ordered UL, UR, LR, LL)."""
    color: tuple[int, int, int] | None = None
    """Reserved for OCR text color (not populated by current tools)."""


class DetectedFontTextItem(BaseModel):
    """Text item with detected font information from `traenslenzor.font_detector.mcp.detect_font`."""

    type: Literal["font_detected"] = "font_detected"
    extractedText: str
    """Raw text from PaddleOCR `rec_texts`."""
    confidence: float
    """PaddleOCR confidence score from `rec_scores`."""
    bbox: list[BBoxPoint]
    """PaddleOCR polygon from `rec_polys` (4 points ordered UL, UR, LR, LL)."""
    detectedFont: str
    """Font family name set by `traenslenzor.font_detector.mcp.detect_font`."""
    font_size: int
    """Font size set by `traenslenzor.font_detector.mcp.detect_font`."""
    color: tuple[int, int, int] | None = None
    """Reserved for OCR text color (not populated by current tools)."""


class TranslatedTextItem(BaseModel):
    """Fully processed text item with translation from `traenslenzor.translator.mcp.translate`."""

    type: Literal["translated"] = "translated"
    extractedText: str
    """Raw text from PaddleOCR `rec_texts`."""
    confidence: float
    """PaddleOCR confidence score from `rec_scores`."""
    bbox: list[BBoxPoint]
    """PaddleOCR polygon from `rec_polys` (4 points ordered UL, UR, LR, LL)."""
    detectedFont: str
    """Font family name set by `traenslenzor.font_detector.mcp.detect_font`."""
    font_size: int
    """Font size set by `traenslenzor.font_detector.mcp.detect_font`."""
    translatedText: str
    """Translated text set by `traenslenzor.translator.mcp.translate`."""
    color: tuple[int, int, int] | None = None
    """Reserved for OCR text color (not populated by current tools)."""


TextItem = Annotated[
    Union[OCRTextItem, DetectedFontTextItem, TranslatedTextItem], Discriminator("type")
]


class ExtractedDocument(BaseModel):
    """Deskewed document metadata from `traenslenzor.text_extractor.mcp.extract_text`."""

    id: str
    """File id for the deskewed image uploaded in `traenslenzor.text_extractor.mcp.extract_text`."""
    transformation_matrix: list[list[float]]
    """3x3 transformation matrix from OpenCV's getPerspectiveTransform used by `traenslenzor.image_renderer.mcp_server.replace_text` to transform rendered text back to original image space."""
    documentCoordinates: list[BBoxPoint]
    """Document polygon for the deskewed image (currently empty in `traenslenzor.text_extractor.mcp.extract_text`)."""


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
