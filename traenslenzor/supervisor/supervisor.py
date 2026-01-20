import logging
from typing import Any, Callable, Optional, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent,
    wrap_tool_call,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.supervisor.llm import llm
from traenslenzor.supervisor.prompt import context_aware_prompt
from traenslenzor.supervisor.state import SupervisorState, ToolCall
from traenslenzor.supervisor.tools.mcp import format_tool_label
from traenslenzor.supervisor.tools.tools import get_tools

logger = logging.getLogger(__name__)


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
    tool_label = format_tool_label(request.tool_call["name"])

    async def _set_active_tool(label: str | None) -> None:
        if not session_id:
            return
        try:
            await SessionClient.update(session_id, lambda s: setattr(s, "activeTool", label))
        except Exception:
            logger.exception("Failed to update active tool status.")

    await _set_active_tool(tool_label)

    try:
        result = await handler(request)  # type: ignore
    finally:
        await _set_active_tool(None)
    if isinstance(result, ToolMessage):
        result = Command(
            update={
                "messages": [result],
            }
        )
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
            llm,
            tools=tools,
            checkpointer=MemorySaver(),
            state_schema=SupervisorState,
            # debug= True, # enhanced logging
            middleware={initialize_session, context_aware_prompt, wrap_tools},  # type: ignore
        )


def select_final_message(messages: list[BaseMessage]) -> BaseMessage:
    if messages and isinstance(messages[-1], AIMessage):
        return messages[-1]
    logger.warning("Supervisor ended without a final AI message. Returning fallback summary.")
    return AIMessage(
        content="Workflow completed. Please check the results and respond with necessary adjustments"
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
    session_id = result.get("session_id", None)

    return (select_final_message(messages), session_id)
