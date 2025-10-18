from typing import Literal

from langchain.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END

from traenslenzor.supervisor.state import MessagesState

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    previous_messages: list[AnyMessage] = [
        SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        )
    ] + state["messages"]
    return {
        "messages": [llm_with_tools.invoke(previous_messages)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage)
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:  # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    assert isinstance(last_message, AIMessage)
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we stop (reply to the user)
    return END
