from __future__ import annotations

import pickle
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import streamlit as st

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import (
    ExtractedDocument,
    SessionState,
    SuperResolvedDocument,
)

T = TypeVar("T")

DEFAULT_SESSION_PICKLE_PATH = Path(".data/streamlit_session_state.pkl")
_PENDING_RESTORE_KEY = "_pending_session_restore"
_AUTO_RESTORE_FLAG = "_auto_restore_attempted"


def _collect_session_file_manifest(session: SessionState) -> dict[str, str]:
    manifest: dict[str, str] = {}
    if session.rawDocumentId:
        manifest[session.rawDocumentId] = "raw.png"
    extracted = session.extractedDocument
    if extracted:
        manifest[extracted.id] = "extracted.png"
        if extracted.mapXYId:
            manifest[extracted.mapXYId] = "map_xy.npy"
    if session.renderedDocumentId:
        manifest[session.renderedDocumentId] = "rendered.png"
    superres = session.superResolvedDocument
    if superres:
        manifest[superres.id] = "superres.png"
    return manifest


def _build_session_payload(session: SessionState) -> dict[str, Any]:
    payload: dict[str, Any] = session.model_dump()
    payload.pop("rawDocumentId", None)
    payload.pop("renderedDocumentId", None)

    extracted = payload.get("extractedDocument")
    if isinstance(extracted, dict):
        extracted.pop("id", None)
        extracted.pop("mapXYId", None)

    superres = payload.get("superResolvedDocument")
    if isinstance(superres, dict):
        superres.pop("id", None)
        superres.pop("sourceId", None)

    return payload


def get_session_id() -> str | None:
    return cast(str | None, st.session_state.get("last_session_id"))


def set_session_id(value: str | None) -> None:
    if st.session_state.get("last_session_id") != value:
        st.session_state.pop("cached_session", None)
        st.session_state.pop("cached_progress", None)
        st.session_state.pop("extracted_image_id", None)
        st.session_state.pop("extracted_image", None)
    st.session_state["last_session_id"] = value


def has_user_interaction(default_assistant_message: str) -> bool:
    history = cast(list[dict[str, str]], st.session_state.get("history", []))
    if len(history) > 1:
        return True
    if history and history[0].get("content") != default_assistant_message:
        return True
    if get_session_id():
        return True
    if st.session_state.get("pending_supervisor_future"):
        return True
    return False


def build_export_payload(
    session: SessionState,
    *,
    run_async: Callable[[Awaitable[T]], T],
) -> tuple[dict[str, object], list[str]]:
    payload: dict[str, object] = {}
    skipped_files: list[str] = []

    payload["session_state"] = _build_session_payload(session)

    files: dict[str, dict[str, object]] = {}
    manifest = _collect_session_file_manifest(session)
    for file_id, name in manifest.items():
        try:
            data = run_async(FileClient.get_raw_bytes(file_id))
        except Exception:
            data = None
        if not data:
            skipped_files.append(file_id)
            continue
        files[name] = {"name": name, "bytes": data}
    payload["files"] = files

    if not files:
        skipped_files.append("no_files")

    return payload, skipped_files


def write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def read_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def queue_restore_payload(
    payload: dict[str, object],
) -> None:
    st.session_state[_PENDING_RESTORE_KEY] = {
        "payload": payload,
    }
    st.rerun(scope="app")


def apply_restore_payload(
    payload: dict[str, object],
    run_async: Callable[[Awaitable[T]], T],
) -> None:
    session_state = payload.get("session_state") if isinstance(payload, dict) else None
    files_state = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(session_state, dict):
        st.session_state["session_restore_error"] = "Missing session_state payload."
        return
    if not isinstance(files_state, dict):
        st.session_state["session_restore_error"] = "Missing files payload."
        return

    def upload_named(name: str) -> str | None:
        entry = files_state.get(name)
        if not isinstance(entry, dict):
            return None
        data = entry.get("bytes")
        if not isinstance(data, (bytes, bytearray)):
            return None
        try:
            return run_async(FileClient.put_bytes(name, bytes(data)))
        except Exception:
            return None

    file_ids: dict[str, str | None] = {
        "raw": upload_named("raw.png"),
        "extracted": upload_named("extracted.png"),
        "rendered": upload_named("rendered.png"),
        "superres": upload_named("superres.png"),
        "map_xy": upload_named("map_xy.npy"),
    }

    if file_ids["raw"] is None and file_ids["extracted"] is None:
        st.session_state["session_restore_error"] = "Missing required document files."
        return

    extracted_meta = session_state.get("extractedDocument")
    superres_meta = session_state.get("superResolvedDocument")

    if extracted_meta is not None and file_ids["extracted"] is None:
        st.session_state["session_restore_error"] = "Missing extracted document file."
        return
    if isinstance(extracted_meta, dict):
        if extracted_meta.get("mapXYShape") is not None and file_ids["map_xy"] is None:
            st.session_state["session_restore_error"] = "Missing map_xy file."
            return

    if superres_meta is not None and file_ids["superres"] is None:
        st.session_state["session_restore_error"] = "Missing super-res file."
        return

    restored = SessionState(
        rawDocumentId=file_ids["raw"],
        renderedDocumentId=file_ids["rendered"],
        text=session_state.get("text"),
        language=session_state.get("language"),
        class_probabilities=session_state.get("class_probabilities"),
    )

    if isinstance(extracted_meta, dict) and file_ids["extracted"]:
        restored.extractedDocument = ExtractedDocument(
            id=file_ids["extracted"],
            documentCoordinates=extracted_meta.get("documentCoordinates") or [],
            mapXYId=file_ids["map_xy"],
            mapXYShape=extracted_meta.get("mapXYShape"),
        )

    if isinstance(superres_meta, dict) and file_ids["superres"]:
        source = superres_meta.get("source") or "deskewed"
        if source not in ("raw", "rendered", "deskewed"):
            source = "deskewed"
        source_id = None
        if source == "raw":
            source_id = file_ids["raw"]
        elif source == "rendered":
            source_id = file_ids["rendered"]
        else:
            source_id = file_ids["extracted"]
        if source_id is None:
            st.session_state["session_restore_error"] = "Missing super-res source file."
            return
        restored.superResolvedDocument = SuperResolvedDocument(
            id=file_ids["superres"],
            sourceId=source_id,
            source=source,
            model=superres_meta.get("model", "unknown"),
            scale=superres_meta.get("scale", 1),
        )

    try:
        target_session_id = run_async(SessionClient.create(restored))
    except Exception as exc:
        st.session_state["session_restore_error"] = f"Failed to restore session: {exc}"
        return

    set_session_id(str(target_session_id))
    st.session_state.pop("last_session_signature", None)
    st.session_state.pop("cached_session", None)
    st.session_state.pop("cached_progress", None)
    if "session_restore_error" not in st.session_state:
        st.session_state["session_restore_notice"] = "Session restored from pickle."


def apply_pending_restore(*, run_async: Callable[[Awaitable[T]], T]) -> bool:
    pending = st.session_state.pop(_PENDING_RESTORE_KEY, None)
    if not isinstance(pending, dict):
        return False
    payload = pending.get("payload")
    if not isinstance(payload, dict):
        st.session_state["session_restore_error"] = "Invalid restore payload."
        return False
    apply_restore_payload(payload, run_async=run_async)
    st.session_state[_AUTO_RESTORE_FLAG] = True
    return True


def maybe_auto_restore_from_pickle(
    *,
    default_assistant_message: str,
    run_async: Callable[[Awaitable[T]], T],
) -> None:
    if st.session_state.get(_AUTO_RESTORE_FLAG):
        return
    st.session_state[_AUTO_RESTORE_FLAG] = True
    if has_user_interaction(default_assistant_message):
        return

    import_path = Path(
        st.session_state.get("session_import_path", DEFAULT_SESSION_PICKLE_PATH)
    ).expanduser()
    if not import_path.exists():
        return

    try:
        payload = read_pickle(import_path)
    except Exception as exc:
        st.session_state["session_restore_error"] = f"Failed to load pickle: {exc}"
        return
    if not isinstance(payload, dict):
        st.session_state["session_restore_error"] = "Pickle payload must be a dict."
        return

    apply_restore_payload(
        payload,
        run_async=run_async,
    )


def apply_session_deletions(
    *,
    session: SessionState,
    session_id: str,
    delete_raw: bool,
    delete_extracted: bool,
    delete_rendered: bool,
    delete_superres: bool,
    delete_text: bool,
    delete_classification: bool,
    delete_language: bool,
    delete_ui_history: bool,
    delete_image_cache: bool,
    delete_files: bool,
    run_async: Callable[[Awaitable[T]], T],
    default_assistant_message: str,
) -> None:
    file_ids: list[str] = []

    if delete_raw and session.rawDocumentId:
        file_ids.append(session.rawDocumentId)
    if delete_extracted and session.extractedDocument:
        extracted = session.extractedDocument
        file_ids.append(extracted.id)
        if extracted.mapXYId:
            file_ids.append(extracted.mapXYId)
    if delete_rendered and session.renderedDocumentId:
        file_ids.append(session.renderedDocumentId)
    if delete_superres and session.superResolvedDocument:
        file_ids.append(session.superResolvedDocument.id)

    def update_session(state: SessionState) -> None:
        if delete_raw:
            state.rawDocumentId = None
        if delete_extracted:
            state.extractedDocument = None
        if delete_rendered:
            state.renderedDocumentId = None
        if delete_superres:
            state.superResolvedDocument = None
        if delete_text:
            state.text = None
        if delete_classification:
            state.class_probabilities = None
        if delete_language:
            state.language = None

    run_async(SessionClient.update(session_id, update_session))

    if delete_files:
        for file_id in file_ids:
            run_async(FileClient.rem(file_id))

    if delete_ui_history:
        st.session_state["history"] = [{"role": "assistant", "content": default_assistant_message}]
    if delete_image_cache:
        st.session_state.pop("image_cache", None)
        st.session_state.pop("extracted_image_id", None)
        st.session_state.pop("extracted_image", None)

    st.session_state.pop("cached_session", None)
    st.session_state.pop("cached_progress", None)
