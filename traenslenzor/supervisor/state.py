from typing import List, Optional

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


class State(TypedDict):
    messages: List[AnyMessage]
    doc_loaded: bool
    language: Optional[str]
    allowed_tools: List[str]
    next_node: str
