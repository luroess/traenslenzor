from typing import List, Literal, Optional

from typing_extensions import TypedDict


class PlanStep(TypedDict):
    action: Literal[
        "request_user_input",
        "language_setter",
        "document_loader",
        "document_translator",
        "document_classifier",
        "font_extractor",
        "document_image_renderer",
        "finish",
    ]
    rationale: Optional[str]
    input: Optional[str]


class State(TypedDict):
    question: str
    steps: List[PlanStep]
    last_observation: Optional[str]
    answer: Optional[str]
    step_count: int
    doc_loaded: bool
    language: Optional[str]


def initialize_state(question: str) -> State:
    return {
        "question": question,
        "steps": [],
        "last_observation": None,
        "answer": None,
        "step_count": 0,
        "doc_loaded": False,
        "language": None,
    }


# <|start_header_id|>system<|end_header_id|>\n\n
# You are a careful reasoner using tools.
# Think step by step **internally**.
# When ready, emit a JSON PlanStep with one of actions: dict_keys(['language_setter', 'document_loader', 'document_translator', 'document_classifier', 'font_extractor', 'document_image_renderer'])  finish.
# ever reveal your rationale to the user.
# If finish, put the final answer in input.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: hi\nLastObservation: None\nSo far you did 0 steps.
# Propose the next PlanStep as JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
