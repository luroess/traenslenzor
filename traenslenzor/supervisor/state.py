sfrom typing import Optional

from langchain.agents.middleware.types import AgentState


class SupervisorState(AgentState):
    language: Optional[str]
    original_document: Optional[str]


def initialize_supervisor_state() -> SupervisorState:
    return SupervisorState(language=None, original_document=None, messages=[])
