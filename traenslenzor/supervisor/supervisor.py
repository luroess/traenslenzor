import asyncio
import logging
from typing import cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from traenslenzor.supervisor.llm import llm
from traenslenzor.supervisor.state import SupervisorState
from traenslenzor.supervisor.tools.tools import get_tools

logger = logging.getLogger(__name__)


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    state = cast(SupervisorState, request.state)
    mem = format_memory(state.get("memory", {}))
    logger.info("Currently:")
    logger.info(
        f"""  - {f"the user has selected the language {state['language']}" if state.get("language", False) else "the user has no language selected"}"""
    )
    logger.info("The llm has stored the following in memory")
    logger.info(mem)
    logger.info("last message: %s", state.get("messages", [])[-1])

    return f"""
    Task:
        You are an image translation assistant.
        Your goal is to turn an image with text in one language into an image in another language.
        Do not imitate actions or describe intended tool use.
        Whenever an action is required, output solely the tool invocation as JSON, with no additional text.
        Set important information like ids in memory so you can recall them later.

    Steps:
        1. Ask the user to provide an image or document. Do not assume any file exists.
        2. Ask the user for the target language and save it.
        3. Extract all text from the image and detect font type, size, and color. Show the text to the user for verification.
        4. Translate the text into the target language, preserving formatting where possible.
        5. Render the translated text on the image, matching the original font and style. Let the user review and request adjustments.
    
    Context:
        - {
        f"the user has selected the language {state['language']}"
        if state.get("language", False)
        else "the user has no language selected"
    }
        - The languages available for translation are "German", "English" and "French"
    Memory:\n{mem}
    """
    # 3. Offer preprocessing options (crop, rotate, enhance). Only continue after the user confirms the image is ready.


def format_memory(memory: dict[str, str]) -> str:
    default = "        No memory entries yet."
    mem = default
    for key, value in memory.items():
        entry = f"        - '{key}': '{value}'"
        if mem == default:
            mem = entry
        else:
            mem = "\n" + entry
    return mem


class Supervisor:
    def __init__(self, tools) -> None:
        self.agent = create_agent(
            llm,
            tools=tools,
            checkpointer=MemorySaver(),
            state_schema=SupervisorState,
            # debug= True, # enhanced logging
            middleware={context_aware_prompt},  # type: ignore
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
