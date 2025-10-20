import logging
from typing import cast

from langchain.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from traenslenzor.supervisor.llm import execute_tools, route, supervisor_call
from traenslenzor.supervisor.state import State
from traenslenzor.supervisor.tools import policy

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self) -> None:
        agent_builder = StateGraph(State)

        agent_builder.add_node("policy", policy)
        agent_builder.add_node("supervisor", supervisor_call)
        agent_builder.add_node("tools", execute_tools)

        agent_builder.set_entry_point("policy")
        agent_builder.add_conditional_edges(
            "policy", route, {"supervisor": "supervisor", "policy": "policy", "END": END}
        )
        agent_builder.add_conditional_edges(
            "supervisor", route, {"tools": "tools", "policy": "policy", "END": END}
        )
        agent_builder.add_conditional_edges("tools", route, {"policy": "policy", "END": END})

        self.agent: CompiledStateGraph[State] = cast(
            CompiledStateGraph[State], agent_builder.compile()
        )

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

        initial_state: State = {
            "messages": [HumanMessage(user_input)],
            "doc_loaded": False,
            "language": None,
            "allowed_tools": [],
            "next_node": "policy",
        }

        result = supervisor.agent.invoke(initial_state)
        for m in result["messages"]:
            m.pretty_print()
