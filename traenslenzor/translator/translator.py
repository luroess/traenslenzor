import json
import logging

from ollama import Client

from traenslenzor.file_server.session_state import TextItem
from traenslenzor.supervisor.config import settings

logger = logging.getLogger(__name__)

client = Client(host=settings.llm.ollama_url)


def translate(text: TextItem, lang: str) -> TextItem:
    """Translate a single text item into the target language.
    This uses the configured LLM to translate the ``extractedText`` field of the
    provided :class:`TextItem` into the specified language. If the model cannot
    sensibly translate the input, it should return the original text unchanged.
    Args:
        text: The text item containing the source text in ``extractedText``.
        lang: The target language code or name to translate the text into.
    Returns:
        A new :class:`TextItem` copied from the input, with ``translatedText``
        set to the translation (or the original text if translation is not
        possible).
    """
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

    new_item = text.model_copy()
    new_item.translatedText = response.message.content
    return new_item


def translate_all(texts: list[TextItem], lang: str) -> list[TextItem]:
    """
    Translate a list of TextItem instances into the target language using batch processing.

    This function first attempts to translate all items in a single LLM call by sending a
    JSON array of the `extractedText` values and expecting a JSON array of translated
    strings in the same order. The response may be wrapped in Markdown code fences, which
    are stripped before JSON parsing.

    If the batch response cannot be parsed as JSON, does not return a list matching the
    input length, or if any error occurs while calling the LLM, the function logs the
    issue and falls back to sequential translation by calling `translate` for each item.

    Args:
        texts: A list of TextItem objects whose `extractedText` fields will be translated.
        lang: The ISO language code or language name to translate the texts into.

    Returns:
        A list of TextItem objects, preserving the input order, where each item has its
        `translatedText` field populated with the corresponding translation (or original
        text if translation was not possible).
    """
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

            if isinstance(translated_texts, list) and len(translated_texts) == len(texts):
                results = []
                for i, item in enumerate(texts):
                    new_item = item.model_copy()
                    new_item.translatedText = translated_texts[i]
                    results.append(new_item)
                return results
            else:
                logger.warning(
                    "Batch translation returned invalid format or length. Falling back to sequential."
                )

    except json.JSONDecodeError as e:
        logger.error(f"Batch translation JSON decode failed: {e}. Falling back to sequential.")
    except Exception as e:
        logger.exception(f"Unexpected error during batch translation: {e}. Falling back to sequential.")

    # Fallback to sequential
    results = []
    for item in texts:
        results.append(translate(item, lang))
    return results
