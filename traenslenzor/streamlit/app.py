from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import streamlit as st
from PIL import ImageDraw

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import (
    BBoxPoint,
    SessionProgress,
    SessionState,
    TextItem,
    initialize_session,
)
from traenslenzor.logger import setup_logger
from traenslenzor.streamlit.prompt_presets import PromptPreset, get_prompt_presets
from traenslenzor.supervisor.supervisor import run as run_supervisor

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from PIL.Image import Image
    from streamlit.elements.widgets.chat import ChatInputValue

    from traenslenzor.file_server.session_state import BBoxPoint, TextItem


T = TypeVar("T")

_STAGES = ["raw", "extracted", "rendered"]
ENABLE_SESSION_POLLING = False
ENABLE_SUPERVISOR_POLLING = True
_SUPERVISOR_POLL_INTERVAL_SECONDS = 2

DEFAULT_ASSISTANT_MESSAGE = (
    "Document Assistant Ready! I can help you with document operations. Please provide a document:"
)


def _default_history() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]


def _ensure_defaults() -> None:
    """Ensure base session state keys exist."""
    st.session_state.setdefault("history", _default_history())
    st.session_state.setdefault("last_session_id", None)


def _get_history() -> list[dict[str, str]]:
    return cast(list[dict[str, str]], st.session_state["history"])


def _get_session_id() -> str | None:
    return cast(str | None, st.session_state.get("last_session_id"))


def _set_session_id(value: str | None) -> None:
    if st.session_state.get("last_session_id") != value:
        st.session_state.pop("cached_session", None)
        st.session_state.pop("cached_progress", None)
    st.session_state["last_session_id"] = value


_ensure_defaults()

setup_logger()
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")


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

    def submit(self, coro: Awaitable[T]) -> Future[T]:
        """Schedule a coroutine on the runner's loop and return a future.

        Args:
            coro (Awaitable[T]): Coroutine to execute.

        Returns:
            Future[T]: Concurrent future tracking the coroutine result.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


@st.cache_resource
def _get_async_runner() -> AsyncRunner:
    """Return a cached async runner for the Streamlit session."""
    return AsyncRunner()


def _run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine on the cached async runner."""
    return _get_async_runner().run(coro)


def _submit_async(coro: Awaitable[T]) -> Future[T]:
    """Schedule a coroutine on the cached async runner."""
    return _get_async_runner().submit(coro)


def _prompt_presets_key() -> str:
    nonce = st.session_state.get("prompt_presets_nonce", 0)
    return f"prompt_presets__{nonce}"


def _render_prompt_presets(presets: list[PromptPreset]) -> str | None:
    """Render quick prompt presets and return the chosen prompt."""
    if not presets:
        return None
    options = [preset.label for preset in presets]

    st.caption("Click a prompt to send it.")
    selection = st.pills("Quick prompts", options, key=_prompt_presets_key())
    if selection is None:
        return None
    st.session_state["prompt_presets_last"] = selection
    return next(preset.prompt for preset in presets if preset.label == selection)


def _get_pending_supervisor_future() -> Future[tuple["BaseMessage", str | None]] | None:
    """Return the pending supervisor future, if any."""
    return cast(
        Future[tuple["BaseMessage", str | None]] | None,
        st.session_state.get("pending_supervisor_future"),
    )


def _is_supervisor_running() -> bool:
    """Check whether the supervisor is still running."""
    return bool(st.session_state.get("supervisor_running", False))


def _get_image_cache() -> dict[str, Image]:
    cache = st.session_state.setdefault("image_cache", {})
    return cast(dict[str, Image], cache)


def _consume_pending_supervisor() -> bool:
    """Append completed supervisor output to chat history.

    Returns:
        bool: True if a completed supervisor future was consumed.
    """
    future = _get_pending_supervisor_future()
    if future is None or not future.done():
        return False
    try:
        msg, session_id = future.result()
    except Exception as exc:
        _get_history().append({"role": "assistant", "content": f"Supervisor failed: {exc!s}"})
        st.toast("Supervisor failed.", icon="⚠️")
    else:
        _set_session_id(session_id)
        st.session_state.pop("cached_session", None)  # Invalidate to fetch fresh
        st.session_state.pop("cached_progress", None)
        _get_history().append({"role": "assistant", "content": msg.content})
        st.toast("Supervisor finished.", icon="✅")
    st.session_state.pop("pending_supervisor_future", None)
    st.session_state["supervisor_running"] = False
    return True


def _start_supervisor_run(llm_input: str) -> None:
    """Start a supervisor run in the background if none is running.

    Args:
        llm_input (str): User prompt for the supervisor.
    """
    if _is_supervisor_running():
        return
    session_id = _ensure_session_id()
    future = _submit_async(run_supervisor(llm_input, session_id=session_id))
    st.session_state["pending_supervisor_future"] = future
    st.session_state["supervisor_running"] = True


def _get_active_session(force_refresh: bool = False) -> SessionState | None:
    """Get session state, optionally forcing a fresh fetch from backend."""
    session_id = _get_session_id()
    if not session_id:
        st.session_state.pop("cached_session", None)
        return None

    # Use cache unless forced to refresh
    if not force_refresh:
        cached = st.session_state.get("cached_session")
        if cached is not None:
            return cast(SessionState, cached)

    try:
        session = _run_async(SessionClient.get(session_id))
        st.session_state["cached_session"] = session
        return session
    except Exception:
        return None


def _get_session_progress(force_refresh: bool = False) -> SessionProgress | None:
    session_id = _get_session_id()
    if not session_id:
        st.session_state.pop("cached_progress", None)
        return None

    if not force_refresh:
        cached = st.session_state.get("cached_progress")
        if cached is not None:
            return cast(SessionProgress, cached)

    try:
        progress = _run_async(SessionClient.get_progress(session_id))
        st.session_state["cached_progress"] = progress
        return progress
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


def _render_session_overview(
    session: SessionState | None, progress: SessionProgress | None
) -> None:
    session_id = _get_session_id()
    if not session_id:
        st.caption("No active session yet.")
        return

    if session is None:
        st.caption("Session state unavailable.")
        return

    if progress is None:
        st.caption("Session progress unavailable.")
        return

    text_count, translated_count, font_count = _count_text_items(session.text)

    done_count = progress.completed_steps
    total_steps = progress.total_steps
    progress_steps = [(step.label, step.done, step.detail) for step in progress.steps]

    # Header with session ID only
    with st.container(border=True):
        st.metric("Session", _short_session_id(session_id))

    # Progress bar and step checklist
    progress_value = done_count / total_steps if total_steps else 0.0
    st.progress(progress_value, text=f"{done_count}/{total_steps} complete")
    for label, done, detail in progress_steps:
        suffix = f" — {detail}" if detail else ""
        st.checkbox(f"{label}{suffix}", value=done, disabled=True)

    # Session details expander
    with st.expander("Session details", expanded=False):
        # Documents section
        st.markdown("**Documents**")
        doc_cols = st.columns(3)
        doc_cols[0].caption("Raw")
        doc_cols[0].code(session.rawDocumentId[:13] + "..." if session.rawDocumentId else "—")
        doc_cols[1].caption("Extracted")
        extracted_id = session.extractedDocument.id if session.extractedDocument else None
        doc_cols[1].code(extracted_id[:13] + "..." if extracted_id else "—")
        doc_cols[2].caption("Rendered")
        doc_cols[2].code(
            session.renderedDocumentId[:13] + "..." if session.renderedDocumentId else "—"
        )

        if session.extractedDocument and session.extractedDocument.documentCoordinates:
            st.caption(
                f"Extracted document coordinates: {len(session.extractedDocument.documentCoordinates)} points"
            )

        # Text items section
        if session.text:
            st.markdown("**Text items**")
            item_cols = st.columns(3)
            item_cols[0].metric("Items", text_count)
            item_cols[1].metric("Translated", f"{translated_count}/{text_count}")
            item_cols[2].metric("Fonts", f"{font_count}/{text_count}")

            # Text items table
            rows = [
                {
                    "text": item.extractedText[:40]
                    + ("..." if len(item.extractedText) > 40 else ""),
                    "translated": (item.translatedText or "")[:40],
                    "confidence": f"{item.confidence:.3f}",
                }
                for item in session.text
            ]
            st.dataframe(rows, width="stretch", hide_index=True)

        # Classification section
        st.markdown("**Classification**")
        if session.class_probabilities:
            sorted_probs = sorted(
                session.class_probabilities.items(), key=lambda x: x[1], reverse=True
            )
            for label, prob in sorted_probs[:3]:
                st.write(f"- {label}: {prob:.1%}")
        else:
            st.caption("No class probabilities available yet.")

        with st.expander("Full session state JSON", expanded=False):
            st.json(session.model_dump(), expanded=False)


def _maybe_rerun_if_supervisor_done() -> None:
    """Trigger a full app rerun when the background supervisor finishes."""
    future = _get_pending_supervisor_future()
    if future is not None and future.done():
        st.rerun(scope="app")


def _render_session_sidebar_content() -> None:
    """Render the sidebar's Session panel using cached or refreshed state."""
    session = _get_active_session(force_refresh=_is_supervisor_running())
    progress = _get_session_progress(force_refresh=_is_supervisor_running())

    st.subheader("Session")
    _render_session_overview(session, progress)


@st.fragment(run_every=_SUPERVISOR_POLL_INTERVAL_SECONDS)
def _render_sidebar_fragment() -> None:
    """Sidebar fragment that polls for session updates while supervisor runs."""
    _maybe_rerun_if_supervisor_done()
    _render_session_sidebar_content()


@st.fragment(run_every=_SUPERVISOR_POLL_INTERVAL_SECONDS)
def _render_supervisor_watchdog() -> None:
    """Poll for supervisor completion and trigger an app rerun."""
    _maybe_rerun_if_supervisor_done()


def _render_sidebar() -> None:
    """Render sidebar with optional polling."""
    if ENABLE_SESSION_POLLING:
        _render_sidebar_fragment()
        return

    if ENABLE_SUPERVISOR_POLLING and _is_supervisor_running():
        _render_supervisor_watchdog()

    _render_session_sidebar_content()


def _render_chat(history: list[dict[str, str]]) -> None:
    st.title("TrÄenslÄnzÖr 0815 Döküment Äsißtänt")
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if _is_supervisor_running():
        with st.chat_message("assistant"):
            st.status("TrÄenslönzing...", state="running", expanded=False)


def _draw_document_outline(image: Image, document_coordinates: list[BBoxPoint]) -> Image:
    """Draw document polygon coordinates on top of the image.

    Args:
        image (Image): Base image to annotate.
        document_coordinates (list[BBoxPoint]): Polygon points in image space.

    Returns:
        Image: Annotated image with the document outline.
    """
    if not document_coordinates:
        return image

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    points = [(int(round(point.x)), int(round(point.y))) for point in document_coordinates]
    if len(points) < 2:
        return annotated

    draw.line(points + [points[0]], fill=(0, 200, 0), width=3)
    corner_radius = 2
    for x, y in points:
        draw.ellipse(
            (x - corner_radius, y - corner_radius, x + corner_radius, y + corner_radius),
            outline=(255, 0, 0),
            width=2,
        )
    return annotated


def fetch_image(
    session: SessionState,
    stage: Literal["raw", "extracted", "rendered"],
) -> tuple[Image | None, str | None, str | None]:
    """Fetch the requested stage image and apply overlays as needed.

    Args:
        session: Active session state.
        stage: Stage to fetch.

    Returns:
        Image, file ID, and optional failure reason.
    """
    file_id: str | None = None
    document_coordinates: list[BBoxPoint] | None = None
    image: Image | None = None
    failure_reason: str | None = None

    match stage:
        case "raw":
            file_id = session.rawDocumentId
            if not file_id:
                failure_reason = "No raw document."
        case "extracted":
            extracted = session.extractedDocument
            if extracted is None:
                failure_reason = "No extracted document."
            else:
                file_id = extracted.id
                document_coordinates = extracted.documentCoordinates
        case "rendered":
            file_id = session.renderedDocumentId
            if not file_id:
                failure_reason = "Rendered document id is missing."
    if failure_reason is None and file_id:
        cache = _get_image_cache()
        if (cached := cache.get(file_id)) is not None:
            image = cached
        else:
            try:
                image = _run_async(FileClient.get_image(file_id))
            except Exception:
                logger.exception("Failed to fetch image for stage %s (file_id=%s)", stage, file_id)
                failure_reason = "Failed to fetch image."
            if image is not None:
                image.load()
                cache[file_id] = image
            elif failure_reason is None:
                failure_reason = "Image not found."

    if image is not None and stage == "extracted" and document_coordinates:
        image = _draw_document_outline(image, document_coordinates)

    return image, file_id, failure_reason


def _ensure_session_id() -> str:
    if session_id := _get_session_id():
        return session_id
    session_id = _run_async(SessionClient.create(initialize_session()))
    _set_session_id(session_id)
    return session_id


def _render_image(session: SessionState | None) -> None:
    if session is None:
        return

    tabs = st.tabs([stage.title() for stage in _STAGES])
    for stage, tab in zip(_STAGES, tabs):
        with tab:
            img, file_id, reason = fetch_image(session, stage)
            if img is not None:
                st.caption(f"{stage.title()} document image ({file_id}).")
                st.image(img, width="stretch")
            elif reason:
                st.caption(reason)


def _collect_prompt() -> str | ChatInputValue | None:
    presets = get_prompt_presets()
    preset_prompt = _render_prompt_presets(presets)
    prompt = st.chat_input(
        "Paste an image or say something",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
    )
    if prompt is None and preset_prompt:
        return preset_prompt
    return prompt


def _handle_prompt(prompt: str | ChatInputValue) -> None:
    user_text = prompt if isinstance(prompt, str) else prompt.text
    uploaded_files = [] if isinstance(prompt, str) else list(prompt.files)

    # Reset preset selection on any prompt
    st.session_state["prompt_presets_nonce"] = st.session_state.get("prompt_presets_nonce", 0) + 1
    st.session_state.pop("prompt_presets_last", None)

    # Handle file upload
    if uploaded_files:
        if len(uploaded_files) > 1:
            st.warning("Only the first file will be processed.")
        uploaded = uploaded_files[0]
        session_id = _ensure_session_id()
        file_id = _run_async(FileClient.put_bytes(uploaded.name, uploaded.getvalue()))
        if file_id:
            _run_async(
                SessionClient.update(session_id, lambda s: setattr(s, "rawDocumentId", file_id))
            )
            st.session_state.pop("cached_session", None)
            st.session_state.pop("cached_progress", None)
            _get_history().append({"role": "user", "content": f"[Image pasted] {uploaded.name}"})

    # Build LLM input
    if user_text:
        _get_history().append({"role": "user", "content": user_text})
        _start_supervisor_run(user_text)
    elif uploaded_files:
        _start_supervisor_run(
            "I just uploaded an image. Acknowledge receipt and ask what I want to do next. "
            "Do not call any tools yet."
        )

    st.rerun()


def _render_layout(session: SessionState | None) -> None:
    chat_col, img_col = st.columns([1, 1], gap="large")
    with chat_col:
        _render_chat(_get_history())
    with img_col:
        _render_image(session)


def main() -> None:
    # Handle completed supervisor runs
    if _consume_pending_supervisor():
        st.rerun()

    with st.sidebar:
        _render_sidebar()

    # Main layout uses cached session (fragment handles live updates)
    active_session = _get_active_session()
    _render_layout(active_session)

    prompt = _collect_prompt()
    if prompt is not None:
        _handle_prompt(prompt)


main()
