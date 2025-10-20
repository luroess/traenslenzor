from langchain.tools import tool

from traenslenzor.supervisor.state import State


@tool
def set_language(language: str) -> str:
    """Sets the language for translation."""
    return f"Language set to {language}"


@tool
def load_document(filepath: str) -> str:
    """Loads a document from the given filepath."""
    return f"Document loaded from {filepath}"


@tool
def preprocess_document(content: str) -> str:
    """Preprocesses the document content."""
    return f"Document preprocessed: {content}"


@tool
def translate_document(content: str, target_language: str) -> str:
    """Translates the document content to the target language."""
    return f"Document translated to {target_language}: {content}"


@tool
def classify_document(content: str) -> str:
    """Classifies the document content."""
    return f"Document classified: {content}"


TOOLS = {
    "set_language": set_language,
    "load_document": load_document,
    "preprocess_document": preprocess_document,
    "translate_document": translate_document,
    "classify_document": classify_document,
}


def policy(state: State) -> State:
    """Policy: derives allowed_tools from state and sets next_node"""
    if state.get("doc_loaded", False):
        allowed = [
            "preprocess_document",
            "translate_document",
            "classify_document",
            "load_document",
            "set_language",
        ]
    else:
        allowed = ["set_language", "load_document"]

    return {**state, "allowed_tools": allowed, "next_node": "supervisor"}
