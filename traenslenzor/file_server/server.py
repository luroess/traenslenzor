import asyncio
import uuid
from io import BytesIO
from typing import Dict, Tuple, get_args

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server

from traenslenzor.file_server.session_state import (
    HasFontInfo,
    HasTranslation,
    ProgressStage,
    SessionProgress,
    SessionProgressStep,
    SessionState,
)

app = FastAPI(title="File Server")

ADDRESS = "127.0.0.1"
PORT = 8001
STORE: Dict[str, Tuple[bytes, str, str]] = {}
STATE: Dict[str, SessionState] = {}
PROGRESS_STAGE_ORDER: list[str] = list(get_args(ProgressStage))


@app.post("/files")
async def upload_file(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    file_id = str(uuid.uuid4())
    STORE[file_id] = (
        data,
        file.filename or "file",
        file.content_type or "application/octet-stream",
    )
    return {
        "id": file_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(data),
    }


@app.get("/files/{file_id}")
async def get_file(file_id: str):
    item = STORE.get(file_id)
    if not item:
        raise HTTPException(status_code=404, detail="File not found")
    data, filename, content_type = item

    return StreamingResponse(
        BytesIO(data),
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(data)),
        },
    )


@app.delete("/files/{file_id}", status_code=204)
async def delete_file(file_id: str):
    if file_id not in STORE:
        raise HTTPException(status_code=404, detail="File not found")
    del STORE[file_id]
    return


@app.post("/sessions", response_model=Dict[str, str])
async def create_session(session: SessionState):
    session_id = str(uuid.uuid4())
    STATE[session_id] = session
    return {"id": session_id}


@app.get("/sessions/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    session = STATE.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/sessions/{session_id}/progress", response_model=SessionProgress)
async def get_session_progress(session_id: str):
    session = STATE.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _compute_session_progress(session_id, session)


def _compute_session_progress(session_id: str, session: SessionState) -> SessionProgress:
    text_items = session.text or []
    text_count = len(text_items)
    translated = sum(isinstance(item, HasTranslation) for item in text_items)
    fonted = sum(isinstance(item, HasFontInfo) for item in text_items)

    def detail(count: int) -> str | None:
        return None if not text_count else f"{count}/{text_count}"

    class_probs = session.class_probabilities or {}
    top_class = max(class_probs.items(), key=lambda item: item[1])[0] if class_probs else None

    steps_data = [
        ("Document loaded", bool(session.rawDocumentId or session.extractedDocument), None),
        ("Language detected", bool(session.language), session.language),
        ("Text extracted", text_count > 0, f"{text_count} items" if text_count else None),
        ("Translated", text_count > 0 and translated == text_count, detail(translated)),
        ("Font detected", text_count > 0 and fonted == text_count, detail(fonted)),
        (
            "Classified",
            bool(class_probs),
            f"{top_class} ({class_probs[top_class]:.0%})" if top_class else None,
        ),
        ("Rendered", bool(session.renderedDocumentId), None),
    ]

    steps = [
        SessionProgressStep(label=label, done=done, detail=info) for label, done, info in steps_data
    ]
    fallback_stage = PROGRESS_STAGE_ORDER[-1] if PROGRESS_STAGE_ORDER else "rendering"
    stage = next(
        (PROGRESS_STAGE_ORDER[idx] for idx, step in enumerate(steps) if not step.done),
        fallback_stage,
    )

    return SessionProgress(
        session_id=session_id,
        stage=stage,
        completed_steps=sum(step.done for step in steps),
        total_steps=len(steps),
        steps=steps,
    )


@app.put("/sessions/{session_id}", response_model=SessionState)
async def replace_session(session_id: str, session: SessionState):
    if session_id not in STATE:
        raise HTTPException(status_code=404, detail="Session not found")
    STATE[session_id] = session
    return session


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if session_id not in STATE:
        raise HTTPException(status_code=404, detail="Session not found")
    del STATE[session_id]
    return


async def run():
    config = Config(
        app=app,
        host=ADDRESS,
        port=PORT,
        reload=False,
        log_level="info",
    )
    server = Server(config)
    print("Launched the fileserver")
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run())
