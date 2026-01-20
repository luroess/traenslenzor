import asyncio

import pytest

from traenslenzor.app import session_state_tools as sst
from traenslenzor.file_server.session_state import (
    ExtractedDocument,
    SessionState,
    SuperResolvedDocument,
)


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.rerun_called = False
        self.rerun_scope = None

    def rerun(self, *, scope: str | None = None) -> None:
        self.rerun_called = True
        self.rerun_scope = scope


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture()
def fake_st(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setattr(sst, "st", fake)
    return fake


def test_build_export_payload_strips_ids_and_collects_files(fake_st, monkeypatch):
    session = SessionState(
        rawDocumentId="raw-id",
        renderedDocumentId="rendered-id",
        language="en",
        class_probabilities={"x": 0.5},
        extractedDocument=ExtractedDocument(
            id="extracted-id",
            documentCoordinates=[],
            mapXYId="mapxy-id",
            mapXYShape=(1, 2, 2),
            mapXYZId="mapxyz-id",
            mapXYZShape=(1, 2, 3),
        ),
        superResolvedDocument=SuperResolvedDocument(
            id="superres-id",
            sourceId="extracted-id",
            source="deskewed",
            model="m",
            scale=2,
        ),
    )

    async def fake_get_raw_bytes(file_id: str):
        return f"data-{file_id}".encode()

    monkeypatch.setattr(sst.FileClient, "get_raw_bytes", fake_get_raw_bytes)

    payload, skipped = sst.build_export_payload(session, run_async=run_async)

    assert skipped == []
    assert "session_state" in payload
    assert "files" in payload

    state = payload["session_state"]
    assert "rawDocumentId" not in state
    assert "renderedDocumentId" not in state
    assert "activeTool" not in state
    assert "extractedDocument" in state
    assert "superResolvedDocument" in state
    assert "id" not in state["extractedDocument"]
    assert "mapXYId" not in state["extractedDocument"]
    assert "mapXYZId" not in state["extractedDocument"]
    assert "id" not in state["superResolvedDocument"]
    assert "sourceId" not in state["superResolvedDocument"]

    files = payload["files"]
    assert set(files.keys()) == {
        "raw.png",
        "extracted.png",
        "map_xy.npy",
        "map_xyz.npy",
        "rendered.png",
        "superres.png",
    }
    assert files["raw.png"]["bytes"].startswith(b"data-")


def test_build_export_payload_reports_missing_files(fake_st, monkeypatch):
    session = SessionState(rawDocumentId="raw-id")

    async def fake_get_raw_bytes(file_id: str):
        return None

    monkeypatch.setattr(sst.FileClient, "get_raw_bytes", fake_get_raw_bytes)

    payload, skipped = sst.build_export_payload(session, run_async=run_async)
    assert payload["files"] == {}
    assert "no_files" in skipped


def test_queue_restore_payload_sets_pending_and_reruns(fake_st):
    payload = {"session_state": {}, "files": {}}
    sst.queue_restore_payload(payload)
    assert fake_st.rerun_called is True
    assert fake_st.session_state.get(sst._PENDING_RESTORE_KEY) == {"payload": payload}


def test_apply_pending_restore_invokes_restore(fake_st, monkeypatch):
    called = {}

    def fake_apply(payload, run_async):
        called["payload"] = payload
        called["run_async"] = run_async

    monkeypatch.setattr(sst, "apply_restore_payload", fake_apply)
    fake_st.session_state[sst._PENDING_RESTORE_KEY] = {
        "payload": {"session_state": {}, "files": {}}
    }

    result = sst.apply_pending_restore(run_async=run_async)

    assert result is True
    assert called["payload"] == {"session_state": {}, "files": {}}
    assert fake_st.session_state.get(sst._AUTO_RESTORE_FLAG) is True


def test_apply_restore_payload_requires_files(fake_st):
    sst.apply_restore_payload({"session_state": {}}, run_async=run_async)
    assert fake_st.session_state["session_restore_error"] == "Missing files payload."


def test_apply_restore_payload_missing_extracted_file_sets_error(fake_st, monkeypatch):
    payload = {
        "session_state": {
            "extractedDocument": {
                "documentCoordinates": [],
                "mapXYShape": None,
                "mapXYZShape": None,
            }
        },
        "files": {"raw.png": {"name": "raw.png", "bytes": b"raw"}},
    }

    async def fake_put_bytes(name: str, data: bytes):
        return f"{name}-id"

    async def fake_create(session):
        raise AssertionError("SessionClient.create should not be called")

    monkeypatch.setattr(sst.FileClient, "put_bytes", fake_put_bytes)
    monkeypatch.setattr(sst.SessionClient, "create", fake_create)

    sst.apply_restore_payload(payload, run_async=run_async)

    assert fake_st.session_state["session_restore_error"] == "Missing extracted document file."


def test_apply_restore_payload_recreates_session(fake_st, monkeypatch):
    payload = {
        "session_state": {
            "text": None,
            "language": "en",
            "class_probabilities": {"x": 0.5},
            "extractedDocument": {
                "documentCoordinates": [],
                "mapXYShape": (1, 2, 2),
                "mapXYZShape": (1, 2, 3),
            },
            "superResolvedDocument": {"source": "raw", "model": "m", "scale": 2},
        },
        "files": {
            "raw.png": {"name": "raw.png", "bytes": b"raw"},
            "extracted.png": {"name": "extracted.png", "bytes": b"ex"},
            "map_xy.npy": {"name": "map_xy.npy", "bytes": b"xy"},
            "map_xyz.npy": {"name": "map_xyz.npy", "bytes": b"xyz"},
            "rendered.png": {"name": "rendered.png", "bytes": b"rend"},
            "superres.png": {"name": "superres.png", "bytes": b"sr"},
        },
    }

    async def fake_put_bytes(name: str, data: bytes):
        return f"{name}-id"

    created = {}

    async def fake_create(session):
        created["session"] = session
        return "new-session-id"

    monkeypatch.setattr(sst.FileClient, "put_bytes", fake_put_bytes)
    monkeypatch.setattr(sst.SessionClient, "create", fake_create)

    sst.apply_restore_payload(payload, run_async=run_async)

    restored = created["session"]
    assert restored.rawDocumentId == "raw.png-id"
    assert restored.renderedDocumentId == "rendered.png-id"
    assert restored.extractedDocument.id == "extracted.png-id"
    assert restored.extractedDocument.mapXYId == "map_xy.npy-id"
    assert restored.extractedDocument.mapXYZId == "map_xyz.npy-id"
    assert restored.superResolvedDocument.id == "superres.png-id"
    assert restored.superResolvedDocument.sourceId == "raw.png-id"
    assert restored.superResolvedDocument.source == "raw"
    assert fake_st.session_state["last_session_id"] == "new-session-id"
    assert fake_st.session_state["session_restore_notice"] == "Session restored from pickle."


def test_apply_session_deletions_removes_files_and_state(fake_st, monkeypatch):
    session = SessionState(
        rawDocumentId="raw-id",
        renderedDocumentId="rendered-id",
        extractedDocument=ExtractedDocument(
            id="extracted-id",
            documentCoordinates=[],
            mapXYId="mapxy-id",
            mapXYShape=(1, 2, 2),
            mapXYZId="mapxyz-id",
            mapXYZShape=(1, 2, 3),
        ),
        superResolvedDocument=SuperResolvedDocument(
            id="superres-id",
            sourceId="extracted-id",
            source="deskewed",
            model="m",
            scale=2,
        ),
    )

    removed = []

    async def fake_rem(file_id: str):
        removed.append(file_id)
        return True

    async def fake_update(session_id: str, updater):
        updater(session)
        return session

    monkeypatch.setattr(sst.FileClient, "rem", fake_rem)
    monkeypatch.setattr(sst.SessionClient, "update", fake_update)

    fake_st.session_state["history"] = [{"role": "assistant", "content": "old"}]
    fake_st.session_state["image_cache"] = {"x": "y"}
    fake_st.session_state["extracted_image_id"] = "x"
    fake_st.session_state["extracted_image"] = "y"

    sst.apply_session_deletions(
        session=session,
        session_id="sess-1",
        delete_raw=True,
        delete_extracted=True,
        delete_rendered=True,
        delete_superres=True,
        delete_text=False,
        delete_classification=False,
        delete_language=False,
        delete_ui_history=True,
        delete_image_cache=True,
        delete_files=True,
        run_async=run_async,
        default_assistant_message="default",
    )

    assert set(removed) == {
        "raw-id",
        "extracted-id",
        "mapxy-id",
        "mapxyz-id",
        "rendered-id",
        "superres-id",
    }
    assert fake_st.session_state["history"] == [{"role": "assistant", "content": "default"}]
    assert "image_cache" not in fake_st.session_state
    assert "extracted_image_id" not in fake_st.session_state
    assert "extracted_image" not in fake_st.session_state
