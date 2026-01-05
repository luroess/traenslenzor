import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState

logger = logging.getLogger(__name__)


@tool
async def set_target_language(language: str, runtime: ToolRuntime) -> Command:
    """Sets the translation target language so it can be recalled later.
    Args:
        language (str): the language to remember
    """
    logger.info(f"Setting language to {language}")

    def update_language(session: SessionState):
        session.language = language

    await SessionClient.update(runtime.state["session_id"], update_language)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Language set successfully to {language}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
