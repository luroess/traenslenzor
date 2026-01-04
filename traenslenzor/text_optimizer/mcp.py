import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.text_optimizer.text_optimizer import optimize_text as do_optimize_text

ADDRESS = "127.0.0.1"
PORT = 8008
TEXT_OPTIMIZER_PATH = f"http://{ADDRESS}:{PORT}/mcp"

text_optimizer = FastMCP("Text Optimizer")

logger = logging.getLogger(__name__)


@text_optimizer.tool
async def optimize_text(session_id: str, user_suggestion: str) -> str:
    """Translates text.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
        user_suggestions: suggestions from the user on how to optimize the extracted text.
    """
    logger.info("optimizing text", user_suggestion)
    session = await SessionClient.get(session_id)
    if session is None:
        return "Session invalid"

    present_text = [t.extractedText for t in session.text or []]
    optimized = do_optimize_text(present_text, user_suggestion)

    def update_text(session: SessionState):
        if session.text is None:
            logger.error("no text present")
            return
        for text_element, optim in zip(session.text, optimized):
            text_element.extractedText = optim

    updated_session = await SessionClient.update(session_id, update_text)

    return "Successfully optimized text\n" + "\n".join(
        [f"{index}: {text.extractedText}" for index, text in enumerate(updated_session.text or [])]
    )


async def run():
    await text_optimizer.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
