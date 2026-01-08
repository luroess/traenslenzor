# TextItem Lifecycle Design

This document describes the discriminated union design for managing text items through the translation pipeline. The design allows each processing step to be called independently while maintaining type safety.

## Problem Statement

A linear pipeline where OCR → Font Detection → Translation → Rendering creates issues:

- **Translation doesn't need font info** - it only needs `extractedText`
- **Font Detection doesn't need translation** - it only needs bbox and image data
- These operations can happen in parallel after OCR

## Solution: Diamond-Pattern State Machine

Instead of a linear pipeline, we model a diamond pattern:

```
              ┌─────────────┐
              │  OCRTextItem│
              │    (ocr)    │
              └──────┬──────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│FontDetectedItem │    │TranslatedOnlyItem│
│ (font_detected) │    │(translated_only) │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ RenderReadyItem │
          │ (render_ready)  │
          └─────────────────┘
```

## Design Principles

### 1. Composition with Nested Models

Use nested models for optional data groups:

```python
class FontInfo(BaseModel):
    """Font detection results - only present after font detection."""
    detectedFont: str
    font_size: int

class TranslationInfo(BaseModel):
    """Translation results - only present after translation."""
    translatedText: str
```

### 2. Base Model for Common OCR Data

All text items share common OCR data via inheritance:

```python
class OCRBase(BaseModel):
    """Base OCR data present in all text items."""
    extractedText: str
    confidence: float
    bbox: list[BBoxPoint]
    color: tuple[int, int, int] | None = None
```

### 3. Discriminated Union Variants

Each pipeline state is a separate model that inherits from `OCRBase`:

| Type | Discriminator | Has FontInfo | Has TranslationInfo | Use Case |
|------|---------------|--------------|---------------------|----------|
| `OCRTextItem` | `"ocr"` | ❌ | ❌ | After OCR/Layout Detection |
| `FontDetectedItem` | `"font_detected"` | ✅ | ❌ | After Font Detection (no translation yet) |
| `TranslatedOnlyItem` | `"translated_only"` | ❌ | ✅ | After Translation (no font detection yet) |
| `RenderReadyItem` | `"render_ready"` | ✅ | ✅ | Ready for Image Renderer |

### 4. Type-Safe Transitions

Each processing step declares exactly what it accepts and produces:

```python
# Font Detector
def detect_font(item: OCRTextItem | TranslatedOnlyItem) -> HasFontInfo:
    """Adds font info to an item."""
    ...

# Translator  
def translate(item: OCRTextItem | FontDetectedItem) -> HasTranslation:
    """Adds translation to an item."""
    ...

# Renderer
def render(items: list[RenderReadyItem]) -> Image:
    """Requires fully processed items."""
    ...
```

## Type Aliases

For convenience, type aliases are provided:

```python
# Items that need font detection
NeedsFontDetection = OCRTextItem | TranslatedOnlyItem

# Items that need translation
NeedsTranslation = OCRTextItem | FontDetectedItem

# Items that have font info
HasFontInfo = FontDetectedItem | RenderReadyItem

# Items that have translation
HasTranslation = TranslatedOnlyItem | RenderReadyItem
```

## Helper Functions

State transitions are handled by helper functions:

```python
def add_font_info(item: TextItem, font: FontInfo) -> HasFontInfo:
    """Add font detection results to any text item."""
    ...

def add_translation(item: TextItem, translation: TranslationInfo) -> HasTranslation:
    """Add translation results to any text item."""
    ...

def is_render_ready(item: TextItem) -> bool:
    """Check if an item is ready for rendering."""
    ...

def filter_render_ready(items: list[TextItem]) -> list[RenderReadyItem]:
    """Filter a list to only render-ready items."""
    ...

def all_render_ready(items: list[TextItem]) -> bool:
    """Check if all items are ready for rendering."""
    ...
```

## Benefits

1. **Type Safety**: Each processor declares exactly what it needs and produces
2. **Parallel Processing**: Font detection and translation can run in parallel after OCR
3. **Flexibility**: Steps can be called in any order as long as dependencies are met
4. **Clear Requirements**: Renderer explicitly requires `RenderReadyItem`
5. **No Workarounds**: No need for `getattr` hacks with fallback values

## Example Usage

```python
# After OCR
ocr_items: list[OCRTextItem] = run_ocr(image)

# Parallel processing
async def process_item(item: OCRTextItem) -> RenderReadyItem:
    # These can run in parallel
    font_task = detect_font_for_item(item)
    translation_task = translate_item(item)
    
    font_info, translation_info = await asyncio.gather(font_task, translation_task)
    
    # Combine results
    return RenderReadyItem(
        extractedText=item.extractedText,
        confidence=item.confidence,
        bbox=item.bbox,
        color=item.color,
        font=font_info,
        translation=translation_info,
    )

# All items ready for rendering
render_ready_items = await asyncio.gather(*[process_item(i) for i in ocr_items])

# Render with type safety
result = await renderer.replace_text(image, render_ready_items)
```
