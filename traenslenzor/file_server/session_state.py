from typing import Literal

from pydantic import BaseModel


class BBoxPoint(BaseModel):
    """Image-space point used by `traenslenzor.text_extractor.paddleocr.parse_result` and `traenslenzor.text_extractor.mcp.extract_text`."""

    x: float
    """X coordinate in pixels for OCR/document polygons."""
    y: float
    """Y coordinate in pixels for OCR/document polygons."""


class TextItem(BaseModel):
    """OCR text item from `traenslenzor.text_extractor.paddleocr.parse_result`, stored by `traenslenzor.text_extractor.mcp.extract_text`."""

    extractedText: str
    """Raw text from PaddleOCR `rec_texts`."""
    confidence: float
    """PaddleOCR confidence score from `rec_scores`."""
    bbox: list[BBoxPoint]
    """PaddleOCR polygon from `rec_polys` (4 points ordered UL, UR, LR, LL)."""
    detectedFont: str | None = None
    """Font family name set by `traenslenzor.font_detector.mcp.detect_font`."""
    font_size: int | None = None
    """Font size set by `traenslenzor.font_detector.mcp.detect_font`."""
    translatedText: str | None = None
    """Translated text set by `traenslenzor.translator.mcp.translate`."""
    color: tuple[int, int, int] | None = None
    """Reserved for OCR text color (not populated by current tools)."""


class ExtractedDocument(BaseModel):
    """Deskewed document metadata from `traenslenzor.text_extractor.mcp.extract_text`."""

    id: str
    """File id for the deskewed image uploaded in `traenslenzor.text_extractor.mcp.extract_text`."""
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
