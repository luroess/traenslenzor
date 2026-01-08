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
        "content": f"You are an expert translator. Translate the following list of texts into {lang}. Return ONLY a JSON array of strings, where each string is the translation of the corresponding input text. Maintain the order exactly. Do not include any other text or markdown formatting.",
    }

    message = {
        "role": "user",
        "content": json.dumps(input_texts),
    }

    content = None
    try:
        response = client.chat(model=settings.llm.model, messages=[system, message])
        content = response.message.content

        if content:
            # Clean up potential markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            translated_texts = json.loads(content.strip())
            preview = content.strip()
            if len(preview) > 200:
                preview = preview[:200] + "..."

            if isinstance(translated_texts, list) and len(translated_texts) == len(texts):
                results: list[HasTranslation] = []
                for i, item in enumerate(texts):
                    translation_info = TranslationInfo(translatedText=translated_texts[i])
                    results.append(add_translation(item, translation_info))
                return results
            else:
                if not isinstance(translated_texts, list):
                    logger.warning(
                        "Batch translation returned non-list JSON (%s). Preview: %r. Falling back to sequential.",
                        type(translated_texts).__name__,
                        preview,
                    )
                else:
                    logger.warning(
                        "Batch translation returned length mismatch (expected %s, got %s). Preview: %r. Falling back to sequential.",
                        len(texts),
                        len(translated_texts),
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
