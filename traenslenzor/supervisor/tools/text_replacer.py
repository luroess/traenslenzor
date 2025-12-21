import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState

logger = logging.getLogger(__name__)


@tool
async def replace_text(index: int, replacement: str, runtime: ToolRuntime) -> Command:
    """Replace part or sentences upon user request of the translated text.
    Args:
        index (int): index of the string to replace
        replacement (str): the text to replace the current index value with.
    """
    invalid_index = False

    def update_text(session: SessionState):
        if session.text is None or index < 0 and index >= len(session.text):
            global invalid_index
            invalid_index = True
            return

        old_value = session.text[index].extractedText
        logger.info(f"Replacing {index}: '{old_value}' with '{replacement}'")

        session.text[index].extractedText = replacement

    updated_session = await SessionClient.update(runtime.state["session_id"], update_text)

    if invalid_index:
        return command(f"Index {index} does not exist.", runtime)

    if updated_session.text is None:
        return command("Updated session contains no text content", runtime)

    new_text_content = "\n".join(
        [f"{index}: {text.extractedText}" for index, text in enumerate(updated_session.text)]
    )
    return command(f"new text:\n{new_text_content}", runtime)


def command(message: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=message,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
