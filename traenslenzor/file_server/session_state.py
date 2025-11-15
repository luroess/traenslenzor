from typing import List

from pydantic import BaseModel


class BBoxPoint(BaseModel):
    x: float
    y: float


class TextItem(BaseModel):
    extractedText: str
    confidence: float
    bbox: List[BBoxPoint]  # 4 points: UL, UR, LR, LL
    detectedFont: str | None = None
    translatedText: str | None = None


class ExtractedDocument(BaseModel):
    id: str
    documentCoordinates: List[BBoxPoint]  # 4 points: UL, UR, LR, LL


class SessionState(BaseModel):
    rawDocumentId: str | None = None
    extractedDocument: ExtractedDocument | None = None
    renderedDocumentId: str | None = None
    text: List[TextItem] | None = None
    language: str | None = None
