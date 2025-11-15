from langchain.agents.middleware.types import AgentState


class SupervisorState(AgentState):
    session_id: str
