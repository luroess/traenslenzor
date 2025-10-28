import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from traenslenzor.supervisor.llm import llm
from traenslenzor.supervisor.tools import TOOLS
from traenslenzor.supervisor.user_context import UserContext, initialize_context

logger = logging.getLogger(__name__)


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    base = """
        You are a translation tool for images similar to Google Lens.  
        Your task is to guide the user through the translation process step by step:

        1: Document Acquisition  
        Ask the user to provide an image or document to process.

        2: Preprocessing  
        Offer options to preprocess the document (for example cropping, enhancing, or rotating). Continue only after the user confirms they are satisfied with the input.

        3: Text Extraction  
        Extract all text from the image and detect the font used.

        4: Translation  
        Translate the extracted text into the target language specified by the user.

        5: Rendering  
        Recreate the image by rendering the translated text in the original or a matching font style.

        Always keep the workflow clear and confirm each step with the user before moving on to the next.
        Do not stop calling tools until the Rendering step is complete.
    """

    return base


class Supervisor:
    def __init__(self) -> None:
        self.agent = create_agent(
            llm,
            tools=TOOLS,
            checkpointer=MemorySaver(),
            context_schema=UserContext,
            # debug= True, # enhanced logging
            middleware={context_aware_prompt},  # type: ignore
        )


if __name__ == "__main__":
    supervisor = Supervisor()

    # Startup greeting
    print("Document Assistant Ready!")
    print("I can help you with document operations. Please provide a document.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        config: RunnableConfig = {"configurable": {"thread_id": "69"}}
        result = supervisor.agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            context=initialize_context(),
            config=config,
        )
        while interrupts := result.get("__interrupt__"):
            interrupt_value = interrupts[0].value
            print("Agent: ", interrupt_value)
            user_response = input("User: ")
            result = supervisor.agent.invoke(Command(resume=user_response), config=config)

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print("Agent: ", last_message.content)
