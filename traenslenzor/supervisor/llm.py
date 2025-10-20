import requests
from langchain.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from traenslenzor.supervisor.state import State
from traenslenzor.supervisor.tools import TOOLS

try:
    # check ollama server
    requests.get("http://localhost:11434", timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)

llm = ChatOllama(model="llama3.1", temperature=0, seed=69)


def supervisor_call(state: State) -> State:
    """Supervisor call: binds dynamically only the allowed tools"""
    print(f"DEBUG: allowed_tools = {state['allowed_tools']}")
    print(f"DEBUG: doc_loaded = {state.get('doc_loaded', False)}")

    tools = [TOOLS[name] for name in state["allowed_tools"]]
    print(f"DEBUG: actual tools bound = {[tool.name for tool in tools]}")
    llm_with_tools = llm.bind_tools(tools)

    doc_status = "loaded" if state.get("doc_loaded", False) else "not loaded"

    tool_descriptions = {
        "set_language": "- Set language: Change the language setting",
        "load_document": "- Load document: Load a document from a file path",
        "preprocess_document": "- Preprocess document: Clean and prepare the loaded document",
        "translate_document": "- Translate document: Translate the loaded document",
        "classify_document": "- Classify document: Classify the loaded document",
    }

    capability_descriptions = [
        tool_descriptions[name] for name in state["allowed_tools"] if name in tool_descriptions
    ]
    capabilities_text = "\n".join(capability_descriptions)

    previous_messages = [
        SystemMessage(
            content=f"""You are a helpful assistant for document operations.

Document status: {doc_status}

Your current capabilities:
{capabilities_text}

CRITICAL RULES YOU MUST FOLLOW:
1. NEVER call tools to answer questions about what you can do or your capabilities
2. NEVER call tools unless the user EXPLICITLY provides a file path or EXPLICITLY requests an action
3. For questions like "what can you do?", "what tools do you have?", "help", etc. - ONLY respond with text
4. Do NOT make up file paths or parameters - wait for the user to provide them
5. Please tell the user what you can do based on your current capabilities

Examples of when NOT to call tools (respond with text only):
- "What can you do?" → List your capabilities from above


Examples of when TO call tools (user provides specific request):
- "Load /home/user/document.txt" → Call load_document with that path
- "Set language to Spanish" → Call set_language with Spanish"""
        )
    ] + state["messages"]

    ai = llm_with_tools.invoke(previous_messages)

    if hasattr(ai, "tool_calls") and ai.tool_calls:
        next_node = "tools"
    else:
        next_node = "END"

    return {**state, "messages": state["messages"] + [ai], "next_node": next_node}


def execute_tools(state: State) -> State:
    """Execute the tool calls from the last AI message"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls"):
        return {**state, "next_node": "END"}

    tool_messages: list[ToolMessage] = []
    doc_loaded = state.get("doc_loaded", False)
    language = state.get("language")

    for tool_call in last_message.tool_calls:
        tool = TOOLS[tool_call["name"]]
        try:
            tool_result = tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

            if tool_call["name"] == "load_document":
                doc_loaded = True
            elif tool_call["name"] == "set_language":
                language = tool_call["args"].get("language")
        except Exception as e:
            tool_messages.append(
                ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"])
            )

    messages: list[AnyMessage] = state["messages"] + [*tool_messages]

    state_result: State = {
        "messages": messages,
        "doc_loaded": doc_loaded,
        "language": language,
        "allowed_tools": state["allowed_tools"],
        "next_node": "policy",
    }
    return state_result


def route(state: State) -> str:
    """Router: purely deterministic, NO state changes"""
    return state.get("next_node", "policy")
