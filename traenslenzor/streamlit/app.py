from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from traenslenzor.doc_classifier.configs.mcp_config import DocClassifierMCPConfig
from traenslenzor.doc_classifier.configs.path_config import PathConfig
from traenslenzor.doc_classifier.mcp_integration import mcp_server as doc_classifier_mcp_server
from traenslenzor.doc_scanner.visualize import draw_grid_overlay
from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import (
    BBoxPoint,
    HasFontInfo,
    HasTranslation,
    SessionProgress,
    SessionState,
    TextItem,
    initialize_session,
)
from traenslenzor.logger import setup_logger
from traenslenzor.streamlit.prompt_presets import PromptPreset, get_prompt_presets
from traenslenzor.streamlit.session_state_tools import (
    DEFAULT_SESSION_PICKLE_PATH,
    apply_pending_restore,
    apply_session_deletions,
    build_export_payload,
    get_session_id,
    maybe_auto_restore_from_pickle,
    queue_restore_payload,
    read_pickle,
    set_session_id,
    write_pickle,
)
from traenslenzor.supervisor.supervisor import run as run_supervisor

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from streamlit.elements.widgets.chat import ChatInputValue

    from traenslenzor.file_server.session_state import BBoxPoint, TextItem


T = TypeVar("T")

ENABLE_SESSION_POLLING = True
ENABLE_SUPERVISOR_POLLING = True
_SUPERVISOR_POLL_INTERVAL_SECONDS = 10.0

DEFAULT_ASSISTANT_MESSAGE = (
    "Document Assistant Ready! I can help you with document operations. Please provide a document:"
)
_DOC_CLASSIFIER_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / ".configs" / "doc-classifier.toml"
)


@contextmanager
def _allow_large_images(max_pixels: int | None) -> Iterator[None]:
    """Temporarily relax Pillow's decompression bomb limit."""
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None if max_pixels is None else max_pixels
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pixels


def _ensure_defaults() -> None:
    """Ensure base session state keys exist."""
    st.session_state.setdefault(
        "history", [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]
    )
    st.session_state.setdefault("last_session_id", None)


def _get_history() -> list[dict[str, str]]:
    """Return the chat history stored in session state."""
    return cast(list[dict[str, str]], st.session_state["history"])


_ensure_defaults()

setup_logger()
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")


class AsyncRunner:
    """Run async callables on a persistent event loop in a background thread."""

    def __init__(self) -> None:
        """Start a background event loop thread for async tasks."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """Run the event loop forever on the background thread."""
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
    """Return a unique key for prompt preset widgets."""
    nonce = st.session_state.get("prompt_presets_nonce", 0)
    return f"prompt_presets__{nonce}"


def _load_doc_classifier_config() -> DocClassifierMCPConfig:
    """Load the doc-classifier MCP config, creating a default file if needed."""
    config_path = _DOC_CLASSIFIER_CONFIG_PATH
    if not config_path.exists():
        try:
            return DocClassifierMCPConfig(checkpoint_path=None, device="auto")
        except Exception:
            pass

    try:
        return DocClassifierMCPConfig.from_toml(config_path)
    except Exception:
        logger.exception("Failed to load doc-classifier config at %s. Using defaults.", config_path)
        config = DocClassifierMCPConfig(checkpoint_path=None, device="auto")
        return config


def _render_prompt_presets(presets: list[PromptPreset]) -> str | None:
    """Render quick prompt presets and return the chosen prompt."""
    if not presets:
        return None
    options = [preset.label for preset in presets]

    st.caption("Click a prompt to send it.")
    selection = st.pills(
        "Quick prompts", options, key=_prompt_presets_key(), disabled=_is_supervisor_running()
    )
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


def _sync_supervisor_running_flag() -> None:
    """Keep `supervisor_running` consistent with the pending future.

    This is defensive: Streamlit reruns, cache restores, or transient failures can
    leave `supervisor_running` stuck in the wrong state. We recompute it from the
    actual `pending_supervisor_future` to avoid:
    - the UI showing "running" when no background task exists,
    - polling fragments running forever after a restart,
    - stale futures (or exceptions when checking `.done()`) leaving the app in a
      permanently busy state.

    If the future is missing or invalid, we clear it and set the flag to False.
    """
    future = _get_pending_supervisor_future()
    if future is None:
        st.session_state["supervisor_running"] = False
        return
    try:
        st.session_state["supervisor_running"] = not future.done()
    except Exception:
        st.session_state.pop("pending_supervisor_future", None)
        st.session_state["supervisor_running"] = False


def _is_supervisor_running() -> bool:
    """Check whether the supervisor is still running."""
    return bool(st.session_state.get("supervisor_running", False))


def _get_image_cache() -> dict[str, Image.Image]:
    """Return the in-memory image cache used across render stages."""
    cache = st.session_state.setdefault("image_cache", {})
    return cast(dict[str, Image.Image], cache)


def _get_map_xy_cache() -> dict[str, np.ndarray]:
    """Return the cache for map_xy overlays to avoid repeated downloads."""
    cache = st.session_state.setdefault("map_xy_cache", {})
    return cast(dict[str, np.ndarray], cache)


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
        set_session_id(session_id)
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
    session_id = get_session_id()
    if not session_id:
        st.session_state.pop("cached_session", None)
        return None

    # Use cache unless forced to refresh
    if not force_refresh:
        cached = st.session_state.get("cached_session")
        if cached is not None:
            return cast(SessionState, cached)
    else:
        cached = st.session_state.get("cached_session")
        if cached is None:
            cached = None

    try:
        session = _run_async(SessionClient.get(session_id))
        st.session_state["cached_session"] = session
        return session
    except Exception:
        return None


def _get_session_progress(force_refresh: bool = False) -> SessionProgress | None:
    """Fetch session progress, optionally forcing a backend refresh."""
    session_id = get_session_id()
    if not session_id:
        st.session_state.pop("cached_progress", None)
        return None

    if not force_refresh:
        cached = st.session_state.get("cached_progress")
        if cached is not None:
            return cast(SessionProgress, cached)
    else:
        cached = st.session_state.get("cached_progress")
        if cached is None:
            cached = None

    try:
        progress = _run_async(SessionClient.get_progress(session_id))
        st.session_state["cached_progress"] = progress
        return progress
    except Exception:
        return None


def _short_session_id(session_id: str) -> str:
    """Return a short, user-friendly session id for display."""
    if len(session_id) <= 12:
        return session_id
    return f"{session_id[:8]}...{session_id[-4:]}"


def _format_document_corners(points: list[BBoxPoint] | None) -> str:
    """Format document corner points into a compact, multi-line string."""
    if not points or len(points) < 4:
        return "—"
    labels = ("UL", "UR", "LR", "LL")
    formatted = [f"{label}: ({point.x:.0f}, {point.y:.0f})" for label, point in zip(labels, points)]
    return "\n".join(formatted)


def _count_text_items(text_items: list[TextItem] | None) -> tuple[int, int, int]:
    """Count total text items plus translated/font-detected counts."""
    if not text_items:
        return 0, 0, 0
    translated = sum(1 for item in text_items if isinstance(item, HasTranslation))
    fonts = sum(1 for item in text_items if isinstance(item, HasFontInfo))
    return len(text_items), translated, fonts


def _render_session_overview(
    session: SessionState | None, progress: SessionProgress | None
) -> None:
    """Render the session summary panel in the sidebar."""
    session_id = get_session_id()
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
    extracted_doc = session.extractedDocument
    document_corners = extracted_doc.documentCoordinates if extracted_doc else None

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
        st.markdown("**Deskew**")
        deskew_cols = st.columns(3)
        deskew_cols[0].caption("Corners (orig)")
        deskew_cols[0].code(_format_document_corners(document_corners))
        deskew_cols[1].caption("Map XY")
        if extracted_doc and extracted_doc.mapXYId:
            map_shape = extracted_doc.mapXYShape if extracted_doc.mapXYShape else ("?", "?", 2)
            deskew_cols[1].code(f"{map_shape}")
        else:
            deskew_cols[1].code("—")
        deskew_cols[2].caption("Map XY Id")
        if extracted_doc and extracted_doc.mapXYId:
            deskew_cols[2].code(extracted_doc.mapXYId[:13] + "...")
        else:
            deskew_cols[2].code("—")

        # Documents section
        st.markdown("**Documents**")
        doc_cols = st.columns(4)
        doc_cols[0].caption("Raw")
        doc_cols[0].code(session.rawDocumentId[:13] + "..." if session.rawDocumentId else "—")
        doc_cols[1].caption("Extracted")
        extracted_id = extracted_doc.id if extracted_doc else None
        doc_cols[1].code(extracted_id[:13] + "..." if extracted_id else "—")
        doc_cols[2].caption("Rendered")
        doc_cols[2].code(
            session.renderedDocumentId[:13] + "..." if session.renderedDocumentId else "—"
        )
        doc_cols[3].caption("Super-res")
        superres_id = session.superResolvedDocument.id if session.superResolvedDocument else None
        doc_cols[3].code(superres_id[:13] + "..." if superres_id else "—")

        if extracted_doc and extracted_doc.documentCoordinates:
            st.caption(
                f"Extracted document coordinates: {len(extracted_doc.documentCoordinates)} points"
            )

        if session.superResolvedDocument:
            superres = session.superResolvedDocument
            st.caption(
                f"Super-res model: {superres.model} (x{superres.scale}, source={superres.source})"
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
                    "translated": (
                        item.translation.translatedText if hasattr(item, "translation") else ""
                    )[:40],
                    "font_size": item.font.font_size if hasattr(item, "font") else "",
                    "detected_font": item.font.detectedFont if hasattr(item, "font") else "",
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


def _render_session_tools(session: SessionState | None) -> None:
    """Render session import/export and cleanup tools."""
    session_id = get_session_id()
    if session is None:
        st.caption("Session state unavailable.")
    if not session_id:
        st.caption("No active session id.")

    if session is not None and session_id:
        with st.expander("Export session state", expanded=False):
            default_path = DEFAULT_SESSION_PICKLE_PATH
            export_path = st.text_input(
                "Pickle path",
                value=str(st.session_state.get("session_export_path", default_path)),
                key="session_export_path",
            )
            st.caption("Exports session payload + document files (no chat history).")
            if st.button("Save pickle", key="session_export_button"):
                payload, skipped_files = build_export_payload(
                    session,
                    run_async=_run_async,
                )
                if skipped_files:
                    st.error("Export failed: missing files in the file server.")
                else:
                    write_pickle(Path(export_path).expanduser(), payload)
                    st.success(f"Saved session pickle to {export_path}")
    else:
        st.caption("Export requires an active session.")

    with st.expander("Import session state", expanded=False):
        if notice := st.session_state.pop("session_restore_notice", None):
            st.success(notice)
        if error := st.session_state.pop("session_restore_error", None):
            st.error(error)
        default_path = DEFAULT_SESSION_PICKLE_PATH
        import_path = st.text_input(
            "Pickle path",
            value=str(st.session_state.get("session_import_path", default_path)),
            key="session_import_path",
        )
        if st.button("Load pickle", key="session_import_button"):
            try:
                payload = read_pickle(Path(import_path).expanduser())
            except Exception as exc:
                st.error(f"Failed to load pickle: {exc}")
                return
            if not isinstance(payload, dict):
                st.error("Pickle payload must be a dict.")
                return

            queue_restore_payload(
                payload,
            )

    if session is not None and session_id:
        with st.expander("Delete session components", expanded=False):
            col_a, col_b = st.columns(2)
            delete_raw = col_a.checkbox("Raw document", key="delete_raw")
            delete_extracted = col_a.checkbox("Extracted document", key="delete_extracted")
            delete_rendered = col_b.checkbox("Rendered document", key="delete_rendered")
            delete_superres = col_b.checkbox("Super-res document", key="delete_superres")
            delete_text = col_a.checkbox("Text items", key="delete_text")
            delete_classification = col_a.checkbox(
                "Class probabilities", key="delete_classification"
            )
            delete_language = col_b.checkbox("Language", key="delete_language")
            delete_ui_history = col_b.checkbox("Chat history (UI)", key="delete_ui_history")
            delete_image_cache = col_b.checkbox("Image cache (UI)", key="delete_ui_cache")
            delete_files = st.checkbox(
                "Also delete stored files",
                value=True,
                key="delete_files",
            )

            if st.button("Apply deletion", type="primary", key="delete_components_button"):
                apply_session_deletions(
                    session=session,
                    session_id=session_id,
                    delete_raw=delete_raw,
                    delete_extracted=delete_extracted,
                    delete_rendered=delete_rendered,
                    delete_superres=delete_superres,
                    delete_text=delete_text,
                    delete_classification=delete_classification,
                    delete_language=delete_language,
                    delete_ui_history=delete_ui_history,
                    delete_image_cache=delete_image_cache,
                    delete_files=delete_files,
                    run_async=_run_async,
                    default_assistant_message=DEFAULT_ASSISTANT_MESSAGE,
                )
                st.rerun(scope="app")
    else:
        st.caption("Deletion tools require an active session.")


def _session_signature(session: SessionState) -> tuple[object, ...]:
    """Build a signature for key session assets to detect changes."""
    extracted = session.extractedDocument
    superres = session.superResolvedDocument
    return (
        session.rawDocumentId,
        extracted.id if extracted else None,
        extracted.mapXYId if extracted else None,
        session.renderedDocumentId,
        superres.id if superres else None,
    )


def _maybe_rerun_on_session_change(session: SessionState | None) -> None:
    """Rerun the app if key session assets changed to refresh caches/UI.

    This ensures updated document IDs or overlays are reflected immediately.
    """
    if session is None:
        return
    signature = _session_signature(session)
    last_signature = st.session_state.get("last_session_signature")
    if last_signature is None:
        st.session_state["last_session_signature"] = signature
        return
    if signature == last_signature:
        return
    st.session_state["last_session_signature"] = signature
    st.session_state.pop("image_cache", None)
    st.session_state.pop("map_xy_cache", None)
    st.session_state.pop("extracted_image_id", None)
    st.session_state.pop("extracted_image", None)
    st.rerun(scope="app")


def _maybe_rerun_if_supervisor_done() -> None:
    """Trigger a full app rerun when the background supervisor finishes."""
    future = _get_pending_supervisor_future()
    if future is not None and future.done():
        st.rerun(scope="app")


def _render_session_sidebar_content(*, force_refresh: bool) -> SessionState | None:
    """Render the sidebar's Session panel using cached or refreshed state."""
    session = _get_active_session(force_refresh=force_refresh)
    progress = _get_session_progress(force_refresh=force_refresh)

    _render_classifier_checkpoint_selector()
    _render_session_overview(session, progress)
    return session


@st.fragment(run_every=_SUPERVISOR_POLL_INTERVAL_SECONDS)
def _render_sidebar_fragment() -> None:
    """Sidebar fragment that polls for session updates while supervisor runs."""
    _maybe_rerun_if_supervisor_done()
    session = _render_session_sidebar_content(force_refresh=True)
    _maybe_rerun_on_session_change(session)


@st.fragment(run_every=_SUPERVISOR_POLL_INTERVAL_SECONDS)
def _render_supervisor_watchdog() -> None:
    """Poll for supervisor completion and trigger an app rerun."""
    _maybe_rerun_if_supervisor_done()


def _render_classifier_checkpoint_selector() -> None:
    """Render checkpoint selector for the doc-classifier runtime."""
    checkpoints_dir = PathConfig().checkpoints
    checkpoints = sorted(checkpoints_dir.glob("*.ckpt"))

    if not checkpoints:
        st.caption("No classifier checkpoints found in .logs/checkpoints.")
        return

    labels = [path.name for path in checkpoints]

    config = _load_doc_classifier_config()
    current_path = config.checkpoint_path
    current_full = None
    if current_path:
        candidate = Path(current_path)
        current_full = (
            candidate if candidate.is_absolute() else (checkpoints_dir / candidate)
        ).resolve()

    current_index = 0
    if current_full is not None:
        for idx, path in enumerate(checkpoints):
            if path.resolve() == current_full:
                current_index = idx
                break

    selection = st.selectbox(
        "Classifier checkpoint",
        options=labels,
        index=current_index,
        help="Select the Lightning checkpoint used for document classification.",
        key="classifier_checkpoint_select",
    )
    selected_path = checkpoints[labels.index(selection)]
    selected_rel = str(selected_path.name)

    if current_full is not None and current_full.name == selected_rel:
        return

    config.checkpoint_path = Path(selected_rel)
    doc_classifier_mcp_server.reset_runtime()


def _render_sidebar() -> None:
    """Render sidebar with optional polling."""
    st.subheader("Session")
    if ENABLE_SESSION_POLLING and _is_supervisor_running():
        _render_sidebar_fragment()
        return

    if ENABLE_SUPERVISOR_POLLING and _is_supervisor_running():
        _render_supervisor_watchdog()

    _render_session_sidebar_content(force_refresh=False)


def _render_top_right_tools(session: SessionState | None) -> None:
    """Render the top-right popover with session tools."""
    _, right = st.columns([8, 2])
    with right:
        with st.popover("Session tools", use_container_width=True):
            _render_session_tools(session)


def _render_chat(history: list[dict[str, str]]) -> None:
    """Render the chat history and the running indicator."""
    st.title("TrÄenslÄnzÖr 0815 Döküment Äsißtänt")
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if _is_supervisor_running():
        with st.chat_message("assistant"):
            st.status("TrÄenslönzing...", state="running", expanded=False)


def _draw_document_outline(
    image: Image.Image, document_coordinates: list[BBoxPoint]
) -> Image.Image:
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
    stage: Literal["raw", "extracted", "rendered", "superres"],
) -> tuple[Image.Image | None, str | None, str | None]:
    """Fetch the requested stage image and return it with cache support.

    Args:
        session: Active session state.
        stage: Stage to fetch.

    Returns:
        Image, file ID, and optional failure reason.
    """
    file_id: str | None = None
    image: Image.Image | None = None
    failure_reason: str | None = None
    document_coordinates = None

    match stage:
        case "raw":
            file_id = session.rawDocumentId
            if not file_id:
                failure_reason = "No raw document."
            if session.extractedDocument:
                document_coordinates = session.extractedDocument.documentCoordinates
        case "extracted":
            extracted = session.extractedDocument
            if extracted is None:
                failure_reason = "No extracted document."
            else:
                file_id = extracted.id
                cached_id = st.session_state.get("extracted_image_id")
                cached_image = st.session_state.get("extracted_image")
                if cached_id == file_id and isinstance(cached_image, Image.Image):
                    return cached_image, file_id, None
        case "rendered":
            file_id = session.renderedDocumentId
            if not file_id:
                failure_reason = "Rendered document id is missing."
        case "superres":
            superres = session.superResolvedDocument
            if superres is None:
                failure_reason = "No super-resolved document."
            else:
                file_id = superres.id
                if not file_id:
                    failure_reason = "Super-resolved document id is missing."
    if failure_reason is None and file_id:
        cache = _get_image_cache()
        if (cached := cache.get(file_id)) is not None:
            image = cached
        else:
            try:
                max_pixels = 0 if stage == "superres" else 50_000_000
                image = _run_async(FileClient.get_image(file_id, max_pixels=max_pixels))
            except Exception:
                logger.exception("Failed to fetch image for stage %s (file_id=%s)", stage, file_id)
                failure_reason = "Failed to fetch image."
            if image is not None:
                image.load()
                cache[file_id] = image
                if stage == "extracted":
                    st.session_state["extracted_image_id"] = file_id
                    st.session_state["extracted_image"] = image
            elif failure_reason is None:
                failure_reason = "Image not found."

    #
    if image is not None and stage == "raw" and document_coordinates:
        image = _draw_document_outline(image, document_coordinates)

    return image, file_id, failure_reason


def _render_overlay_image(
    session: SessionState,
    *,
    allow_refresh: bool,
) -> tuple[Image.Image | None, str | None, str | None]:
    """Render the raw image with polygon and optional map_xy grid overlays.

    When allow_refresh is False, only cached assets are used to avoid heavy I/O.
    """
    file_id = session.rawDocumentId
    if not file_id:
        return None, None, "No raw document."

    image: Image.Image | None = None
    if not allow_refresh:
        image = _get_image_cache().get(file_id)

    if image is None:
        try:
            image = _run_async(FileClient.get_image(file_id))
        except Exception:
            logger.exception("Failed to fetch raw image for overlay (file_id=%s)", file_id)
            return None, file_id, "Failed to fetch image."

        if image is None:
            return None, file_id, "Image not found."
        image.load()
        _get_image_cache()[file_id] = image

    extracted = session.extractedDocument
    if extracted and extracted.documentCoordinates:
        image = _draw_document_outline(image, extracted.documentCoordinates)

    if extracted and extracted.mapXYId and allow_refresh:
        map_xy = _get_map_xy_cache().get(extracted.mapXYId)
        if map_xy is None:
            try:
                map_xy = _run_async(FileClient.get_numpy_array(extracted.mapXYId))
            except Exception:
                logger.exception("Failed to fetch map_xy overlay (file_id=%s)", extracted.mapXYId)
                map_xy = None
            if map_xy is not None:
                _get_map_xy_cache()[extracted.mapXYId] = map_xy
        if map_xy is not None:
            np_img = np.array(image.convert("RGB"))
            np_img = draw_grid_overlay(np_img, map_xy, step=40)
            image = Image.fromarray(np_img)

    return image, file_id, None


def _ensure_session_id() -> str:
    """Ensure there is an active file-server session id."""
    if session_id := get_session_id():
        return session_id
    session_id = _run_async(SessionClient.create(initialize_session()))
    set_session_id(session_id)
    return session_id


def _render_image_stage(
    stage: str,
    label: str,
    *,
    session: SessionState | None,
    allow_refresh: bool,
    allow_session_refresh: bool,
) -> None:
    """Render a single image stage with optional refresh/caching rules."""
    if allow_session_refresh or session is None:
        session = _get_active_session()
    if session is None:
        st.caption("No active session.")
        return
    if stage == "overlay":
        img, file_id, reason = _render_overlay_image(session, allow_refresh=allow_refresh)
    else:
        img, file_id, reason = fetch_image(session, stage)
    if img is not None:
        st.caption(f"{label} document image ({file_id}).")
        if stage == "superres":
            max_pixels = img.width * img.height
            with _allow_large_images(max_pixels):
                st.image(img, width="stretch")
        else:
            st.image(img, width="stretch")
    elif reason:
        st.caption(reason or "Failed to fetch image")


@st.fragment(run_every=_SUPERVISOR_POLL_INTERVAL_SECONDS)
def _render_image_stage_polling(stage: str, label: str) -> None:
    """Polling fragment that renders an image stage without refresh-heavy work."""
    _render_image_stage(
        stage,
        label,
        session=None,
        allow_refresh=False,
        allow_session_refresh=True,
    )


def _render_image(session: SessionState | None) -> None:
    """Render the image area with stage tabs and polling-aware updates."""
    if session is None:
        return

    tab_specs = [
        ("Raw", "raw"),
        ("Overlay", "overlay"),
        ("Extracted", "extracted"),
        ("Rendered", "rendered"),
        ("Super-res", "superres"),
    ]

    tabs = st.tabs([label for label, _ in tab_specs])
    for (label, stage), tab in zip(tab_specs, tabs):
        with tab:
            if _is_supervisor_running():
                _render_image_stage_polling(stage, label)
            else:
                _render_image_stage(
                    stage,
                    label,
                    session=session,
                    allow_refresh=True,
                    allow_session_refresh=False,
                )


def _collect_prompt() -> str | ChatInputValue | None:
    """Collect user input from chat or prompt presets."""
    presets = get_prompt_presets()
    preset_prompt = _render_prompt_presets(presets)
    prompt = st.chat_input(
        "Paste an image or say something",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
        disabled=_is_supervisor_running(),
    )
    if prompt is None and preset_prompt:
        return preset_prompt
    return prompt


def _handle_prompt(prompt: str | ChatInputValue) -> None:
    """Handle user prompt and file uploads, then start supervisor as needed."""
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
        _run_async(SessionClient.prepare_new_doc(session_id))
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
    """Render the main two-column layout (chat + images)."""
    chat_col, img_col = st.columns([1, 1], gap="large")
    with chat_col:
        _render_chat(_get_history())
    with img_col:
        _render_image(session)


def main() -> None:
    """Streamlit app entrypoint: restore state, render UI, handle input."""
    if not apply_pending_restore(run_async=_run_async):
        maybe_auto_restore_from_pickle(
            default_assistant_message=DEFAULT_ASSISTANT_MESSAGE,
            run_async=_run_async,
        )
    _sync_supervisor_running_flag()
    # Handle completed supervisor runs
    if _consume_pending_supervisor():
        st.rerun()

    with st.sidebar:
        _render_sidebar()

    # Main layout uses cached session; sidebar refresh updates the cache.
    active_session = _get_active_session()
    _render_top_right_tools(active_session)
    _render_layout(active_session)

    prompt = _collect_prompt()
    if prompt is not None:
        _handle_prompt(prompt)


main()
