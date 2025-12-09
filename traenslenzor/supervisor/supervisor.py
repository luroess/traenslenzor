import asyncio
import logging
from typing import Callable, Optional, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage
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
from traenslenzor.supervisor.tools.tools import get_tools

logger = logging.getLogger(__name__)


@wrap_tool_call  # type: ignore
async def wrap_tools(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> Command:
    result = await handler(request)  # type: ignore
    if isinstance(result, ToolMessage):
        result = Command(
            update={
                "messages": [result],
            }
        )
    state = cast(SupervisorState, request.state)
    history = state.get("tool_history", []) + [
        ToolCall(request.tool_call["name"], request.tool_call["args"])
    ]
    result.update["tool_history"] = history  # type: ignore

    logger.info("Tool history")
    logger.info(history)

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


async def run():
    tools = await get_tools()
    supervisor = Supervisor(tools)

    # Startup greeting
    print("Document Assistant Ready!")
    print("I can help you with document operations. Please provide a document.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")

    loop = asyncio.get_event_loop()

    while True:
        user_input = await loop.run_in_executor(None, input, "User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        config: RunnableConfig = {"configurable": {"thread_id": "69"}}
        result = await supervisor.agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print("Agent: ", last_message.content)
