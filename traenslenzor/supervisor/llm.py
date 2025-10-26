from pprint import pprint

import requests
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_ollama import ChatOllama

from traenslenzor.supervisor.state import PlanStep, State
from traenslenzor.supervisor.tools import TOOLS, TOOLS_NAME_MAP

try:
    # check ollama server
    requests.get("http://localhost:11434", timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)

llm = ChatOllama(model="llama3.1", temperature=0, seed=69).bind_tools(TOOLS)

SYSTEM = (
    "You are a careful reasoner using tools. "
    "Think step by step **internally**. "
    # "When ready, emit a JSON PlanStep {{action, rationale, output}}" f"with one of actions: {', '.join(TOOLS_NAME_MAP.keys())}, finish. "
    "Never reveal your rationale to the user. If finish, put the final answer in output."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "user",
            "Question: {question}\n"
            "LastObservation: {last_observation}\n"
            "So far you did {step_count} steps. "
            "Propose the next PlanStep as JSON.",
        ),
    ]
)
parser = JsonOutputParser(pydantic_object=PlanStep)


class LoggingRunnable(Runnable):
    def __init__(self, runnable, name=None):
        self.runnable = runnable
        self.name = name or runnable.__class__.__name__

    def invoke(  # type: ignore[override]
        self,
        input,
        config,
        **kwargs,
    ):
        print(f"[{self.name}] Input:", input)
        result = self.runnable(input, config, **kwargs)
        print(f"[{self.name}] Output:", result)
        return result


logger = LoggingRunnable(pprint)
plan_runnable = prompt | llm | parser


# ---- Nodes
def plan(state: State) -> State:
    step = plan_runnable.invoke(
        {
            "question": state["question"],
            "last_observation": state.get("last_observation", None),
            "step_count": state.get("step_count", 0),
        }
    )
    state["steps"].append(step)
    return state


def act(state: State) -> State:
    step = state["steps"][-1]
    print(f"Acting on step: {step}")
    if step["action"] == "finish":
        state["answer"] = step.get("output", "")  # type: ignore[typeddict-item]
        return state
    tool = TOOLS_NAME_MAP[step["action"]]
    obs = tool.invoke(step.get("output", ""))  # type: ignore
    state["last_observation"] = obs
    state["step_count"] = state.get("step_count", 0) + 1
    return state


def should_continue(state: State):
    print(state)
    last = state["steps"][-1]
    print(last)
    if last["action"] == "finish":
        return "finish"
    if state.get("step_count", 0) >= 6:  # hard stop to avoid loops
        # force finish with best effort
        state["answer"] = (
            f"Partial answer (step cap reached). Latest info: {state.get('last_observation', '')}"
        )
        return "finish"
    return "continue"
