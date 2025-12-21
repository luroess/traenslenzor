from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPreset:
    """Prompt preset used by the Streamlit UI."""

    label: str
    prompt: str


def get_prompt_presets() -> list[PromptPreset]:
    """Return prompt presets covering the available supervisor tools.

    Returns:
        list[PromptPreset]: Ordered prompt presets for quick selection.
    """
    return [
        PromptPreset(
            label="Extract text (OCR)",
            prompt="Extract text from the current document.",
        ),
        PromptPreset(
            label="Detect fonts",
            prompt="Detect the fonts and font sizes used in the document.",
        ),
        PromptPreset(
            label="Set target language to English",
            prompt="Set the translation target language to English.",
        ),
        PromptPreset(
            label="Translate text",
            prompt="Translate the extracted text into the target language.",
        ),
        PromptPreset(
            label="Classify document",
            prompt="Classify this document and return the top 3 classes with probabilities.",
        ),
        PromptPreset(
            label="Full pipeline",
            prompt=(
                "Extract the text, detect fonts, translate to English, "
                "and render the translated document image."
            ),
        ),
    ]
