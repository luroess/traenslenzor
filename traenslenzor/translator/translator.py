import asyncio

from ollama import AsyncClient

from traenslenzor.file_server.session_state import (
    HasTranslation,
    TextItem,
    TranslationInfo,
    add_translation,
)
from traenslenzor.supervisor.config import settings


async def translate(text: TextItem, lang: str) -> HasTranslation:
    system = {
        "role": "system",
        "content": "You are an expert translator. Translate the user's question into "
        + lang
        + " in a concise manner. Keep the word count the same as the input text. If you cannot make sense of the input, just give it back without changing anything",
    }
    message = {
        "role": "user",
        "content": f"Only Respond with the translation, or the original if you cannot translate it. Never say anything else, but the translation or the original text. Text to translate to {lang}: \n{text.extractedText}",
    }
    response = await AsyncClient(host=settings.llm.ollama_url).chat(
        model=settings.llm.model, messages=[system, message], think=False
    )

    assert response.message.content is not None

    translation_info = TranslationInfo(translatedText=response.message.content)
    return add_translation(text, translation_info)


async def translate_all(texts: list[TextItem], lang: str) -> list[HasTranslation]:
    return await asyncio.gather(*[translate(item, lang) for item in texts])
