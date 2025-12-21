import logging
from typing import Any, Callable, Optional, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent,
    wrap_tool_call,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.supervisor.llm import get_llm
from traenslenzor.supervisor.prompt import context_aware_prompt
from traenslenzor.supervisor.state import SupervisorState, ToolCall
from traenslenzor.supervisor.tools.tools import get_tools

logger = logging.getLogger(__name__)

_SESSION_TOOLS = {
    "classify_document",
    "detect_font",
    "extract_text",
    "render_image",
    "translate",
}


class LLMIOLogger(BaseCallbackHandler):
    """Log model input/output for debugging purposes."""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        logger.info("LLM input:")
        for message_group in messages:
            for message in message_group:
                logger.info("[%s] %s", message.type, message.content)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        logger.info("LLM output:")
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message is not None:
                    logger.info("[%s] %s", message.type, message.content)
                else:
                    logger.info("%s", generation.text)


@wrap_tool_call  # type: ignore
async def wrap_tools(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> Command:
    state = cast(SupervisorState, request.state)
    session_id = state.get("session_id")
    if session_id:
        tool_args = request.tool_call.get("args", {})
        if request.tool_call.get("name") in _SESSION_TOOLS:
            tool_args["session_id"] = session_id

    result = await handler(request)  # type: ignore
    if isinstance(result, ToolMessage):
        result = Command(
            update={
                "messages": [result],
            }
        )
    state = cast(SupervisorState, request.state)
    result.update["tool_history"] = [ToolCall(request.tool_call["name"], request.tool_call["args"])]  # type: ignore

    logger.info("Tool history")
    logger.info(state.get("tool_history", []))

    return result


@before_agent
async def initialize_session(state: SupervisorState, runtime: Runtime) -> Optional[Command]:
    if "session_id" not in state:
        logger.info("Creating a new session")
        return Command(update={"session_id": await SessionClient.create(SessionState())})
    logger.info("using existing session %s", state["session_id"])
    return None


class Supervisor:
    def __init__(self, tools) -> None:
        self.agent = create_agent(
            get_llm(),
            tools=tools,
            checkpointer=MemorySaver(),
            state_schema=SupervisorState,
            # debug= True, # enhanced logging
            middleware={initialize_session, context_aware_prompt, wrap_tools},  # type: ignore
        )


async def run(user_input: str, session_id: str | None = None) -> tuple[BaseMessage, str | None]:
    tools = await get_tools()
    supervisor = Supervisor(tools)

    config: RunnableConfig = {
        "configurable": {"thread_id": "69"},
        "callbacks": [LLMIOLogger()],
    }
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": user_input}]}
    if session_id:
        payload["session_id"] = session_id
    result = await supervisor.agent.ainvoke(payload, config=config)

    messages = result.get("messages", [])
    resolved_session_id = result.get("session_id", session_id)

    return (messages[-1], resolved_session_id)
