from typing import Optional

from langchain.agents.middleware.types import AgentState


class SupervisorState(AgentState):
    language: Optional[str]
    memory: dict[str, str]


def initialize_supervisor_state() -> SupervisorState:
    return SupervisorState(language=None, memory={}, messages=[])
