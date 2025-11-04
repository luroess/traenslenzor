import asyncio
import logging
from typing import cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from traenslenzor.supervisor.llm import llm
from traenslenzor.supervisor.state import SupervisorState
from traenslenzor.supervisor.tools.tools import get_tools

logger = logging.getLogger(__name__)


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    state = cast(SupervisorState, request.state)
    logging.info("Currently:")
    logging.info(
        f"""  - {f"the user has a document '{state.get('original_document', False)}' loaded" if state.get("original_document", False) else "the user has no document selected"}"""
    )
    logging.info(
        f"""  - {f"the user has selected the language {state['language']}" if state.get("language", False) else "the user has no language selected"}"""
    )

    return f"""
        You are a translation tool for images similar to Google Lens.  
        Your task is to guide the user through the translation process step by step:

        1: Document Acquisition  
        Ask the user to provide an image or document to process.

        2: Language Acquisition
        Set the language in the state via a tool to save it.

        3: Preprocessing  
        Offer options to preprocess the document (for example cropping, enhancing, or rotating). Continue only after the user confirms they are satisfied with the input.

        4: Text Extraction  
        Extract all text from the image and detect the font used.

        5: Translation  
        Translate the extracted text into the target language specified by the user.

        6: Rendering  
        Recreate the image by rendering the translated text in the original or a matching font style.

        Always keep the workflow clear and keep the user informed on what they need to do next.
        Decide what needs to be done next and call the appropriate tool.
        
        Currently:
            - {
        f"the user has a document '{state.get('original_document', False)}' loaded"
        if state.get("original_document", False)
        else "the user has no document selected"
    }
            - {
        f"the user has selected the language {state['language']}"
        if state.get("language", False)
        else "the user has no language selected"
    }
    """


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
        while interrupts := result.get("__interrupt__"):
            interrupt_value = interrupts[0].value
            print("Agent: ", interrupt_value)
            user_response = await loop.run_in_executor(None, input, "User: ")
            result = await supervisor.agent.ainvoke(Command(resume=user_response), config=config)

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print("Agent: ", last_message.content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
