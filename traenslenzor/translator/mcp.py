import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import (
    SessionState,
    TranslationInfo,
    add_translation,
)

ADDRESS = "127.0.0.1"
PORT = 8005
TRANSLATOR_PATH = f"http://{ADDRESS}:{PORT}/mcp"

translator = FastMCP("Translator")

logger = logging.getLogger(__name__)


@translator.tool
async def translate(session_id: str) -> str:
    """Translates text.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    """

    def update_session(session: SessionState):
        if session.text is not None:
            updated_texts = []
            for t in session.text:
                # Create translation info (reversed text as mock translation)
                translation_info = TranslationInfo(translatedText=t.extractedText[::-1])
                # Add translation to the text item (may return TranslatedOnlyItem or RenderReadyItem)
                updated_item = add_translation(t, translation_info)
                updated_texts.append(updated_item)
            session.text = updated_texts

    await SessionClient.update(session_id, update_session)
    return "Translation successful"


async def run():
    await translator.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
