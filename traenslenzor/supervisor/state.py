from dataclasses import dataclass
from typing import Annotated, Any

from langchain.agents import AgentState


@dataclass
class ToolCall:
    tool: str
    args: Any


class SupervisorState(AgentState):
    session_id: str
    tool_history: Annotated[list[ToolCall], lambda left, right: [*left, *right]]
