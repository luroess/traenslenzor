import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from traenslenzor.supervisor.tools.document_loader import document_loader
from traenslenzor.supervisor.tools.mcp import get_mcp_tools

logger = logging.getLogger(__name__)


# 1. Stage
@tool
def set_target_language(language: str, runtime: ToolRuntime) -> Command:
    """Sets the translation target language so it can be recalled later.
    Args:
        language (str): the language to remember
    """
    logger.info(f"Setting language to {language}")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Language set successfully to {language}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "language": language,
        }
    )


@tool
def store_in_memory(key: str, value: str, runtime: ToolRuntime) -> Command:
    """Stores relevant information in memory for later usage.
    Args:
        key (str): the key by which the infromation is stored.
        value (str): the value that will be rememberd.
    """
    logger.info(f"Setting {key} to {value}")
    new_mem = {**runtime.state.get("memory", {}), key: value}

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Remembered",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "memory": new_mem,
        }
    )


# # 2. Stage
# @tool
# def document_preprocessor(document: str) -> str:
#     """Preprocesses the document content."""
#     return f"Document preprocessed: {document}"


# 3. Stage
@tool
def document_classifier(document: str) -> str:
    """Classifies the document content."""
    return f"Document classified: {document}"


@tool
def font_extractor(document: str) -> str:
    """Extracts fonts used in the document content."""
    return f"Fonts extracted from document: {document}"


# 4. Stage
@tool
def document_translator(document: str, target_language: str) -> str:
    """
    Translates the document content to the target language.
    """
    return f"Document translated to {target_language}: {document}"


# 5. Stage
@tool
def document_image_renderer(document: str, format: str) -> str:
    """Renders the document content in the specified format."""
    return f"Document rendered in {format} format: {document}"


async def get_tools():
    mcp_tools = await get_mcp_tools()
    return [
        set_target_language,
        document_loader,
        store_in_memory,
        # document_preprocessor,
        document_translator,
        document_classifier,
        font_extractor,
        document_image_renderer,
        *mcp_tools,
    ]
