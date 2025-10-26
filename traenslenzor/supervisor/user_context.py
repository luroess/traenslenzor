from dataclasses import dataclass
from typing import Optional


@dataclass
class UserContext:
    doc_loaded: bool
    language: Optional[str]


def initialize_context() -> UserContext:
    return UserContext(doc_loaded=False, language=None)
