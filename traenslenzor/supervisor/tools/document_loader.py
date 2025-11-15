import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from traenslenzor.file_server.client import FileClient

logger = logging.getLogger(__name__)


@tool
async def document_loader(filepath: str, runtime: ToolRuntime) -> Command:
    """Loads a document from the given filepath collected from the user."""
    logger.info(f"Trying to load file: '{filepath}'")

    try:
        file_id = await FileClient.put(filepath)
        logger.info(f"Successfully loaded file: '{filepath}'")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "Document loaded successfully. "
                            f"Use document id {file_id} for all subsequent operations."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "original_document": file_id,
            }
        )
    except Exception as e:
        logger.error(f"Failed to load file: '{filepath}'", e)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Document loading failed: invalid path {filepath}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "original_document": None,
            }
        )
