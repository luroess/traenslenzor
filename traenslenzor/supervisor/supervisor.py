import logging

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from traenslenzor.supervisor.llm import act, plan, should_continue
from traenslenzor.supervisor.state import State, initialize_state

logger = logging.getLogger(__name__)
checkpointer = MemorySaver()


class Supervisor:
    def __init__(self) -> None:
        graph = StateGraph(State)
        graph.add_node("plan", plan)
        graph.add_node("act", act)
        graph.set_entry_point("plan")
        graph.add_conditional_edges("plan", should_continue, {"continue": "act", "finish": END})
        graph.add_edge("act", "plan")

        self.agent = graph.compile(checkpointer=checkpointer)

    def render_graph(self, file):
        try:
            with open(file, "wb") as f:
                png = self.agent.get_graph(xray=True).draw_mermaid_png()
                f.write(png)
        except Exception as e:
            logger.error("Failed to render graph: %s", e)


if __name__ == "__main__":
    supervisor = Supervisor()
    supervisor.render_graph(".supervisor_graph.png")

    # Startup greeting
    print("Document Assistant Ready!")
    print("I can help you with document operations. Please provide a document.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        initial: State = initialize_state(user_input)
        config: RunnableConfig = {"configurable": {"thread_id": "69"}}
        result = supervisor.agent.invoke(initial, config=config)  # type: ignore[arg-type]
        while interrupts := result.get("__interrupt__"):
            interrupt_value = interrupts[0].value
            print("Agent: ", interrupt_value)
            user_response = input("User: ")
            result = supervisor.agent.invoke(Command(resume=user_response), config=config)
        print("Agent: ", result["answer"])
