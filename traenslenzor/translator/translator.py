import json
import logging

from ollama import Client

from traenslenzor.file_server.session_state import (
    HasTranslation,
    TextItem,
    TranslationInfo,
    add_translation,
)
from traenslenzor.supervisor.config import settings

logger = logging.getLogger(__name__)

client = Client(host=settings.llm.ollama_url)


def translate(text: TextItem, lang: str) -> HasTranslation:
    """Translates a single TextItem's extractedText into the target language using the configured LLM."""
    system = {
        "role": "system",
        "content": (
            f"You are an expert translator. Translate the user's question into {lang} "
            f"in a concise manner. Keep the translation concise and roughly similar in "
            f"length to the input text when natural, but prioritize accuracy and "
            f"clarity over matching length exactly. If you cannot make sense of the "
            f"input, just give it back without changing anything"
        ),
    }
    message = {
        "role": "user",
        "content": f"Only Respond with the translation, or the original if you cannot translate it. Never say anything else, but the translation or the original text. Text to translate to {lang}: \n{text.extractedText}",
    }
    response = client.chat(model=settings.llm.model, messages=[system, message])

    assert response.message.content is not None

    translation_info = TranslationInfo(translatedText=response.message.content)
    return add_translation(text, translation_info)


def translate_all(texts: list[TextItem], lang: str) -> list[HasTranslation]:
    """Translates a list of TextItems into the target language using batch processing, falling back to sequential translation on failure."""
    if not texts:
        return []

    input_texts = [t.extractedText for t in texts]

    system = {
        "role": "system",
        "content": f"""
            You are an expert translator. Translate the following list of texts into target language: '{lang}'.
            
            You are given existing text in this format:
                1: Dies ist ein beispiel
                2: Mehr Text

            Your task is to **translate the input into the target language**, **while preserving the original structure and numbering**.
            The **number of indices before and after correction must remain the same**.

            Output must follow the same format as the input:
                1: Translated line 1
                2: Translated line 2

            Do not output anything other than the translated text.
        """,
    }

    message = {
        "role": "user",
        "content": "\n".join([f"{i}: {txt}" for i, txt in enumerate(input_texts)]),
    }

    content = None
    try:
        response = client.chat(model=settings.llm.model, messages=[system, message])
        content = response.message.content

        if content:
            # Generate a short preview for error logging
            preview = content.strip()
            if len(preview) > 200:
                preview = preview[:200] + "..."

            if content is None:
                return None
            lines = content.strip().splitlines()

            # Remove numbering and keep clean text lines (optional)
            translated_lines = []
            for line in lines:
                # Expect format "1: corrected text"
                if ":" in line:
                    _, content = line.split(":", 1)
                    translated_lines.append(content.strip())
                else:
                    # fallback if numbering missing
                    translated_lines.append(line.strip())

            if len(translated_lines) == len(texts):
                results: list[HasTranslation] = []
                for i, item in enumerate(texts):
                    translation_info = TranslationInfo(translatedText=translated_lines[i])
                    results.append(add_translation(item, translation_info))
                return results
            else:
                logger.warning(
                    "Batch translation returned length mismatch (expected %s, got %s). Preview: %r. Falling back to sequential.",
                    len(texts),
                    len(translated_lines),
                    preview,
                )

    except json.JSONDecodeError as e:
        preview = (content or "").strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        logger.error(
            "Batch translation JSON decode failed: %s. Preview: %r. Falling back to sequential.",
            e,
            preview,
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error during batch translation: {e}. Falling back to sequential."
        )

    # Fallback to sequential
    results = []
    for item in texts:
        results.append(translate(item, lang))
    return results
