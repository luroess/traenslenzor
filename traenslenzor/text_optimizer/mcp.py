import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.text_optimizer.text_optimizer import optimize_text as do_optimize_text

ADDRESS = "127.0.0.1"
PORT = 8008
TEXT_OPTIMIZER_PATH = f"http://{ADDRESS}:{PORT}/mcp"

feedback_applier = FastMCP("Text Feedback Applier")

logger = logging.getLogger(__name__)


@feedback_applier.tool
async def apply_text_feedback(session_id: str, user_suggestion: str) -> str:
    """Change the translated text in the current context according to the users feedback.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
        user_suggestions: suggestions from the user on how to change the translated text.
    """
    logger.info(f"applying user feedback {user_suggestion}")
    session = await SessionClient.get(session_id)
    if session is None:
        return "Session invalid"

    present_text = []
    if session.text:
        for t in session.text:
            if t.type == "render_ready":
                present_text.append(t.translation.translatedText)
            else:
                logger.error(f"failed to set text for {t}")
    else:
        logger.error("session.text was empty")
        return "no text"

    optimized = do_optimize_text(present_text, user_suggestion)
    if optimized is None:
        return "failed to optimize text"

    def update_text(session: SessionState):
        if session.text is None:
            logger.error("no text present")
            return
        for text_element, optim in zip(session.text, optimized):
            if text_element.type == "render_ready":
                text_element.translation.translatedText = optim
            else:
                logger.error(f"failed to set text for {text_element}")

    await SessionClient.update(session_id, update_text)

    return "Successfully optimized text\n You should rerender the image now."


async def run():
    await feedback_applier.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
