from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from traenslenzor.file_server.client import FileClient


@tool
async def document_loader(filepath: str, runtime: ToolRuntime) -> Command:
    """Loads a document from the given filepath."""

    try:
        file_id = await FileClient.put(filepath)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Document loaded successfully from {filepath}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "original_document": file_id,
            }
        )
    except Exception:
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
