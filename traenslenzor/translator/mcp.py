import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.translator.translator import translate_all

ADDRESS = "127.0.0.1"
PORT = 8005
TRANSLATOR_PATH = f"http://{ADDRESS}:{PORT}/mcp"

translator = FastMCP("Translator")

logger = logging.getLogger(__name__)


@translator.tool
async def translate(session_id: str) -> str:
    """Translates text for the given session ID."""
    try:
        session = await SessionClient.get(session_id)
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        return f"Error: Failed to get session {session_id}"

    if not session.language:
        return "Error: No target language set in session"

    if not session.text:
        return "Error: No text to translate"

    # Perform translation
    translated_items = translate_all(session.text, session.language)

    if not translated_items:
        logger.warning("Translation returned empty list")
        return "Error: Translation returned no items"

    def update_session(s: SessionState):
        s.text = translated_items

    try:
        await SessionClient.update(session_id, update_session)
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        return "Error: Failed to update session"

    return f"Translation to {session.language} successful"


async def run():
    await translator.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
