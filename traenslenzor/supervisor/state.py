from dataclasses import dataclass
from typing import Any

from langchain.agents import AgentState


@dataclass
class ToolCall:
    tool: str
    args: Any


class SupervisorState(AgentState):
    session_id: str
    tool_history: list[ToolCall]
