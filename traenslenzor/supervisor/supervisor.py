import asyncio
import logging
from typing import Callable, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    before_agent,
    dynamic_prompt,
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
from traenslenzor.supervisor.llm import get_llm
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
async def initialize_session(state: AgentState, runtime: Runtime) -> Command:
    logger.info("Creating a new session")
    return Command(update={"session_id": await SessionClient.create(SessionState())})


@dynamic_prompt
async def context_aware_prompt(request: ModelRequest) -> str:
    state = cast(SupervisorState, request.state)
    session_id = state.get("session_id")
    assert session_id is not None
    session = await SessionClient.get(session_id)

    formatted_session = format_session(session_id, session)
    logger.info("Current Session:")
    logger.info(formatted_session)

    return f"""
    Task:
        You are an image translation assistant.
        Your goal is to turn an image with text in one language into an image in another language.
        Do not imitate actions or describe intended tool use.
        Whenever an action is required, output solely the tool invocation as JSON, with no additional text.
        Execute the steps in order as far as possible.

    Steps:
        1. Ask the user to provide an image or document. Do not assume any file exists.
        2. Retrieve the target language to translate the document into FROM THE USER and save it. Do not assume any language.
        3. Extract all text from the image and detect font type, size, and color. Show the text to the user for verification.
        4. Translate the text into the target language, preserving formatting where possible.
        5. Render the translated text on the image, matching the original font and style. Let the user review and request adjustments.
    
    Context:
        {formatted_session}
    """
    # 3. Offer preprocessing options (crop, rotate, enhance). Only continue after the user confirms the image is ready.


def format_session(session_id: str, session: SessionState) -> str:
    return f"""
        - the current session_id is '{session_id}'
        - {f"the user has selected the language {session.language}" if session.language else "the user has no language selected"}
        - {"the user has a document loaded" if session.rawDocumentId else "the user has no document selected"}
    """


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
