import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)


@tool
def request_user_input(prompt: str) -> str:
    """Requests input from the user with the given prompt.
    Args:
        prompt (str): Question or answer to interact with the user.
    """
    logger.info(f"Asking user question a {prompt}")
    return interrupt(prompt)  # type: ignore[no-any-return]


# 1. Stage
@tool
def language_setter(language: str, runtime: ToolRuntime) -> Command:
    """Sets the language so it can be recalled later.
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
def document_loader(filepath: str, runtime: ToolRuntime) -> Command:
    """Loads a document from the given filepath."""
    logger.info(f"Loading document from {filepath}")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Document loaded successfully from {filepath}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "doc_loaded": True,
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
    """Translates the document content to the target language."""
    return f"Document translated to {target_language}: {document}"


# 5. Stage
@tool
def document_image_renderer(document: str, format: str) -> str:
    """Renders the document content in the specified format."""
    return f"Document rendered in {format} format: {document}"


TOOLS = [
    request_user_input,
    language_setter,
    document_loader,
    # document_preprocessor,
    document_translator,
    document_classifier,
    font_extractor,
    document_image_renderer,
]

TOOLS_NAME_MAP = {t.name: t for t in TOOLS}
