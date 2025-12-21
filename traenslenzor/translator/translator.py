import logging

from ollama import Client

from traenslenzor.file_server.session_state import TextItem
from traenslenzor.supervisor.config import settings

logger = logging.getLogger(__name__)


def translate(text: TextItem, lang: str) -> TextItem:
    system = {
        "role": "system",
        "content": f"You are an expert translator. Translate the user's question into {lang} in a concise manner. Keep the word count the same as the input text. If you cannot make sense of the input, just give it back without changing anything",
    }
    message = {
        "role": "user",
        "content": f"Only Respond with the translation, or the original if you cannot translate it. Never say anything else, but the translation or the original text. Text to translate to {lang}: \n{text.extractedText}",
    }
    response = Client().chat(model=settings.llm.model, messages=[system, message])

    assert response.message.content is not None

    new_item = text.model_copy()
    new_item.translatedText = response.message.content
    return new_item


def translate_all(texts: list[TextItem], lang: str) -> list[TextItem]:
    results = []
    for item in texts:
        results.append(translate(item, lang))
    return results
