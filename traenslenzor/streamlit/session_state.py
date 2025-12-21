from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

from traenslenzor.supervisor.llm import LLMBackend

DEFAULT_ASSISTANT_MESSAGE = (
    "Document Assistant Ready! I can help you with document operations. Please provide a document:"
)


def default_history() -> list[dict[str, str]]:
    """Build the initial chat history.

    Returns:
        list[dict[str, str]]: Default chat history entries.
    """
    return [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]


class SessionStateView:
    """Typed wrapper for Streamlit's session state."""

    def __init__(self, state: MutableMapping[str | int, object]) -> None:
        self._state = state

    def ensure_defaults(self, history: list[dict[str, str]] | None = None) -> None:
        """Ensure base chat keys exist in the session state.

        Args:
            history (list[dict[str, str]] | None): Optional initial history.
        """
        if "history" not in self._state:
            self._state["history"] = history if history is not None else default_history()
        if "last_session_id" not in self._state:
            self._state["last_session_id"] = None

    def ensure_llm_defaults(self, model: LLMBackend, auto_pull: bool = False) -> None:
        """Ensure LLM selection keys exist in the session state.

        Args:
            model (LLMBackend): Default model identifier.
            auto_pull (bool): Default auto-pull flag.
        """
        if "llm_model" not in self._state:
            self._state["llm_model"] = model
        if "llm_auto_pull" not in self._state:
            self._state["llm_auto_pull"] = auto_pull

    @property
    def history(self) -> list[dict[str, str]]:
        """Current chat history.

        Returns:
            list[dict[str, str]]: Chat messages with role and content.
        """
        return cast(list[dict[str, str]], self._state["history"])

    @property
    def last_session_id(self) -> str | None:
        """Last active session id.

        Returns:
            str | None: Last session id if present.
        """
        return (
            cast(str | None, self._state["last_session_id"])
            if "last_session_id" in self._state
            else None
        )

    @last_session_id.setter
    def last_session_id(self, value: str | None) -> None:
        self._state["last_session_id"] = value

    @property
    def llm_model(self) -> LLMBackend:
        """Selected LLM model identifier.

        Returns:
            LLMBackend: Model identifier.
        """
        return cast(LLMBackend, self._state["llm_model"])

    @llm_model.setter
    def llm_model(self, value: LLMBackend) -> None:
        self._state["llm_model"] = value

    @property
    def llm_auto_pull(self) -> bool:
        """Whether to auto-pull missing models."""
        return cast(bool, self._state["llm_auto_pull"])

    @llm_auto_pull.setter
    def llm_auto_pull(self, value: bool) -> None:
        self._state["llm_auto_pull"] = value
