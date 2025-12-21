from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Iterator
from typing import TYPE_CHECKING, TypeVar, cast

import streamlit as st
from PIL.Image import Image

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import initialize_session
from traenslenzor.logger import setup_logger
from traenslenzor.streamlit.prompt_presets import PromptPreset, get_prompt_presets
from traenslenzor.streamlit.session_state import SessionStateView
from traenslenzor.supervisor.llm import (
    LLMBackend,
    SupervisorLLMConfig,
    get_llm_settings,
)
from traenslenzor.supervisor.supervisor import run as run_supervisor

if TYPE_CHECKING:
    from streamlit.elements.widgets.chat import ChatInputValue
    from streamlit.runtime.uploaded_file_manager import UploadedFile

    from traenslenzor.file_server.session_state import SessionState, TextItem

state = SessionStateView(st.session_state)
state.ensure_defaults()

setup_logger()
st.set_page_config(layout="wide")

T = TypeVar("T")


# <TODO: AI fix: >
# Without a persistent loop, the cached client gets tied to a closed loop and you hit the “Event loop is closed” error on later prompts.
# Streamlit used asyncio.run(...) on every interaction.
# get_llm() cached a single ChatOllama client.
# That client’s async HTTP stack was bound to the event loop created by the first asyncio.run(...).
# On the next prompt, asyncio.run(...) created a new loop and closed the old one; the cached LLM client still referenced the old loop → RuntimeError: Event loop is closed (exactly what you saw after “send another prompt”).
# Why AsyncRunner fixes it:


# It creates one long‑lived event loop in a background thread.
# All async calls (LLM, file/session client) run on that same loop.
# The cached LLM client stays bound to a live loop across prompts.
class AsyncRunner:
    """Run async callables on a persistent event loop in a background thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Awaitable[T]) -> T:
        """Execute a coroutine on the runner's loop and return the result.

        Args:
            coro (Awaitable[T]): Coroutine to execute.

        Returns:
            T: Result of the coroutine.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


@st.cache_resource
def _get_async_runner() -> AsyncRunner:
    """Return a cached async runner for the Streamlit session."""
    return AsyncRunner()


def _run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine on the cached async runner."""
    return _get_async_runner().run(coro)


def _render_prompt_presets(presets: list[PromptPreset]) -> str | None:
    """Render quick prompt presets and return the chosen prompt."""
    if not presets:
        return None
    options = [preset.label for preset in presets]

    st.caption("Click a prompt to send it.")
    selection = st.pills("Quick prompts", options, key="prompt_presets")
    if selection is None:
        return None
    if selection == st.session_state.get("prompt_presets_last"):
        return None
    st.session_state["prompt_presets_last"] = selection
    return next(preset.prompt for preset in presets if preset.label == selection)


# </TODO: AI fix.>


def _render_llm_controls() -> None:
    settings = get_llm_settings()
    state.ensure_llm_defaults(settings.model, settings.auto_pull)

    st.subheader("LLM")
    if not isinstance(st.session_state.get("llm_model"), LLMBackend):
        st.session_state["llm_model"] = settings.model
    backend_options = tuple(LLMBackend)
    current_backend = state.llm_model

    selected_model = cast(
        LLMBackend,
        st.selectbox(
            "Model",
            options=LLMBackend,
            index=backend_options.index(current_backend),
            format_func=lambda backend: backend.value,
        ),
    )
    st.checkbox("Auto-pull missing model", key="llm_auto_pull")

    state.llm_model = selected_model
    if selected_model != settings.model:
        SupervisorLLMConfig(
            model=selected_model,
            temperature=settings.temperature,
            seed=settings.seed,
            auto_pull=state.llm_auto_pull,
        ).apply()
    elif state.llm_auto_pull != settings.auto_pull:
        settings.model_copy(update={"auto_pull": state.llm_auto_pull}).apply()


def _get_active_session() -> SessionState | None:
    if not (session_id := state.last_session_id):
        return None
    try:
        return _run_async(SessionClient.get(session_id))
    except Exception:
        return None


def _short_session_id(session_id: str) -> str:
    if len(session_id) <= 12:
        return session_id
    return f"{session_id[:8]}...{session_id[-4:]}"


def _count_text_items(text_items: list[TextItem] | None) -> tuple[int, int, int]:
    if not text_items:
        return 0, 0, 0
    translated = sum(1 for item in text_items if item.translatedText)
    fonts = sum(1 for item in text_items if item.detectedFont)
    return len(text_items), translated, fonts


def _class_summary(class_probs: dict[str, float] | None) -> tuple[bool, str | None]:
    if not class_probs:
        return False, None
    top_label, top_prob = max(class_probs.items(), key=lambda item: item[1])
    return True, f"{top_label} ({top_prob:.0%})"


def _progress_summary(count: int, total: int) -> tuple[bool, str | None]:
    if total == 0:
        return False, None
    if count == total:
        return True, f"{count}/{total}"
    return False, f"{count}/{total}"


def _render_session_overview(session: SessionState | None) -> None:
    st.subheader("Session")

    session_id = state.last_session_id
    if not session_id:
        st.caption("No active session yet.")
        return

    if session is None:
        st.caption("Session state unavailable.")
        return

    text_count, translated_count, font_count = _count_text_items(session.text)
    has_document = bool(session.rawDocumentId or session.extractedDocument)
    has_text = text_count > 0
    has_render = bool(session.renderedDocumentId)
    has_language = bool(session.language)

    translation_done, translation_detail = _progress_summary(translated_count, text_count)
    font_done, font_detail = _progress_summary(font_count, text_count)
    class_done, class_detail = _class_summary(session.class_probabilities)

    steps = [
        ("Document loaded", has_document, None),
        ("Text extracted", has_text, f"{text_count} items" if has_text else None),
        ("Translated", translation_done, translation_detail),
        ("Font detected", font_done, font_detail),
        ("Classified", class_done, class_detail),
        ("Rendered", has_render, None),
    ]

    done_count = sum(1 for _, done, _ in steps if done)
    total_steps = len(steps)

    with st.container(border=True):
        meta_left, meta_right = st.columns(2, gap="small")
        meta_left.metric("Session", _short_session_id(session_id))
        meta_right.metric("Language", session.language if has_language else "Not set")

    st.progress(done_count / total_steps, text=f"{done_count}/{total_steps} complete")

    for label, done, detail in steps:
        suffix = f" — {detail}" if detail else ""
        key = f"session_overview_step__{label.lower().replace(' ', '_')}"
        st.session_state[key] = done
        st.checkbox(f"{label}{suffix}", key=key, disabled=True)


def _render_sidebar(session: SessionState | None) -> None:
    with st.sidebar:
        _render_llm_controls()
        _render_session_overview(session)


def chat_stream(text: str) -> Iterator[str]:
    for ch in text:
        yield ch
        time.sleep(0.005)


def fetch_image(session_id: str | None) -> Image | None:
    if not session_id:
        return None
    sess = _run_async(SessionClient.get(session_id))
    file_ref = sess.extractedDocument
    if not file_ref:
        return None
    return _run_async(FileClient.get_image(file_ref.id))


# TODO: make explicit type for session id to avoid str
def _ensure_session_id() -> str:
    if session_id := state.last_session_id:
        return str(session_id)
    session_id = _run_async(SessionClient.create(initialize_session()))
    state.last_session_id = session_id
    return str(session_id)


def _render_chat(history: list[dict[str, str]]) -> None:
    st.title("TrÄenslÄnzÖr 0815 Döküment Äsißtänt")
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write_stream(chat_stream(message["content"]))
            else:
                st.write(message["content"])


def _render_image(session_id: str | None) -> None:
    # TODO: allow rendering images at multiple stages - i.e. raw, deskwewed, translated
    if not session_id:
        return

    if (img := fetch_image(session_id)) is not None:
        st.image(img, width="stretch")


def _collect_prompt() -> str | ChatInputValue | None:
    presets = get_prompt_presets()
    preset_prompt = _render_prompt_presets(presets)
    prompt = st.chat_input(
        "Paste an image or say something",
        accept_file=True,
        # TODO: what about pdf?
        file_type=["png", "jpg", "jpeg"],
    )
    if prompt is None and preset_prompt:
        return preset_prompt
    return prompt


def _handle_prompt(prompt: str | ChatInputValue) -> None:
    if isinstance(prompt, str):
        # in case of preset prompt - TODO: maybe cast it to ChatInputValue to avoid case distinctions
        user_text = prompt
        uploaded_files: list[UploadedFile] = []
    else:
        user_text = prompt.text
        uploaded_files = list(prompt.files)

    uploaded_names: list[str] = []
    if uploaded_files:
        session_id = _ensure_session_id()
        for uploaded_file in uploaded_files:
            data = uploaded_file.getvalue()
            file_id = _run_async(FileClient.put_bytes(uploaded_file.name, data))
            if not file_id:
                continue
            uploaded_names.append(uploaded_file.name)
            existing_session = _run_async(SessionClient.get(session_id))
            session = initialize_session()
            session.rawDocumentId = file_id
            session.language = existing_session.language
            _run_async(SessionClient.put(session_id, session))

    if uploaded_names:
        state.history.append(
            {"role": "user", "content": f"[Image pasted] {', '.join(uploaded_names)}"}
        )

    llm_input: str | None = None
    if user_text:
        state.history.append({"role": "user", "content": user_text})
        llm_input = user_text
    elif uploaded_names:
        state.history.append(
            {
                "role": "assistant",
                "content": "Image received. Tell me what you want to do with it.",
            }
        )

    if llm_input is not None:
        # TODO: always log the direct I / O of the model
        msg, session_id = _run_async(run_supervisor(llm_input))
        state.last_session_id = session_id
        state.history.append({"role": "assistant", "content": msg.content})

    st.rerun()


def _render_layout(session: SessionState | None) -> None:
    chat_col, img_col = st.columns([1, 1], gap="large")
    with chat_col:
        _render_chat(state.history)
    with img_col:
        _render_image(state.last_session_id)


def main() -> None:
    active_session = _get_active_session()
    _render_sidebar(active_session)
    _render_layout(active_session)

    prompt = _collect_prompt()
    if prompt is not None:
        _handle_prompt(prompt)


main()
