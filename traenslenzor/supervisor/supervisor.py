import logging
from typing import cast

from langchain.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from traenslenzor.supervisor.llm import llm_call, should_continue, tool_node
from traenslenzor.supervisor.state import MessagesState

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self) -> None:
        agent_builder = StateGraph(MessagesState)

        agent_builder.add_node("llm_call", llm_call)
        agent_builder.add_node("tool_node", tool_node)

        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        agent_builder.add_edge("tool_node", "llm_call")

        self.agent: CompiledStateGraph[MessagesState] = cast(
            CompiledStateGraph[MessagesState], agent_builder.compile()
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
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        messages = [HumanMessage(content="Add 3 and 4.")]
        state: MessagesState = {"messages": [HumanMessage(user_input)], "llm_calls": 0}
        for m in supervisor.agent.invoke(state)["messages"]:
            m.pretty_print()
