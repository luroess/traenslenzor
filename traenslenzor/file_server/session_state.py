from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Discriminator


class BBoxPoint(BaseModel):
    x: float
    y: float


class OCRTextItem(BaseModel):
    type: Literal["untranslated"] = "untranslated"
    extractedText: str
    confidence: float
    bbox: List[BBoxPoint]  # 4 points: UL, UR, LR, LL
    color: tuple[int, int, int] | None = None  # TODO: FS look if this is available via paddle


class DetectedFontTextItem(BaseModel):
    type: Literal["font_detected"] = "font_detected"
    extractedText: str
    confidence: float
    bbox: List[BBoxPoint]  # 4 points: UL, UR, LR, LL
    detectedFont: str
    font_size: str
    color: tuple[int, int, int] | None = None  # TODO: FS look if this is available via paddle


class TranslatedTextItem(BaseModel):
    type: Literal["translated"] = "translated"
    extractedText: str
    confidence: float
    bbox: List[BBoxPoint]  # 4 points: UL, UR, LR, LL
    detectedFont: str
    font_size: str
    translatedText: str
    color: tuple[int, int, int] | None = None  # TODO: FS look if this is available via paddle


TextItem = Annotated[
    Union[OCRTextItem, DetectedFontTextItem, TranslatedTextItem], Discriminator("type")
]


class ExtractedDocument(BaseModel):
    id: str
    transformation_matrix: List[List[float]]  # 4 points: UL, UR, LR, LL


class SessionState(BaseModel):
    rawDocumentId: str | None = None
    extractedDocument: ExtractedDocument | None = None
    renderedDocumentId: str | None = None
    text: List[TextItem] | None = None
    language: str | None = None
