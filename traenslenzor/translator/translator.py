import asyncio

from ollama import AsyncClient

from traenslenzor.file_server.session_state import TextItem, TranslatedTextItem


async def translate(text: TextItem, lang: str) -> TranslatedTextItem:
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
    response = await AsyncClient().chat(model="mistral:latest", messages=[system, message])

    assert response.message.content is not None

    return TranslatedTextItem(
        translatedText=response.message.content,
        extractedText=text.extractedText,
        confidence=text.confidence,
        bbox=text.bbox,
        color=text.color,
        detectedFont="Arial",
        font_size="16",
    )


async def translate_all(texts: list[TextItem], lang: str) -> list[TranslatedTextItem]:
    return await asyncio.gather(*[asyncio.create_task(translate(item, lang)) for item in texts])
