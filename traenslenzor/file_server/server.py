import asyncio
import uuid
from io import BytesIO
from typing import Dict, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server

app = FastAPI(title="File Server")

ADDRESS = "127.0.0.1"
PORT = 8005
STORE: Dict[str, Tuple[bytes, str, str]] = {}


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
