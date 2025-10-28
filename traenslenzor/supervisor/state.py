from typing import Optional

from langchain.agents.middleware.types import AgentState


class SupervisorState(AgentState):
    doc_loaded: bool
    language: Optional[str]


def initialize_supervisor_state() -> SupervisorState:
    return SupervisorState(doc_loaded=False, language=None, messages=[])
