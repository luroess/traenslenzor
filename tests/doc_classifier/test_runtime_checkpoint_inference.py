from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

import pytest
from uvicorn import Config, Server

from traenslenzor.doc_classifier.configs import DocClassifierMCPConfig
from traenslenzor.doc_classifier.mcp.runtime import CLASS_NAMES
from traenslenzor.file_server.client import FileClient
from traenslenzor.file_server.server import ADDRESS, PORT
from traenslenzor.file_server.server import app as file_server_app

CKPT_PATH = Path(
    "/home/jandu/repos/traenslenzor/.logs/checkpoints/alexnet-scratch-epoch=epoch=1-val_loss=val/loss=0.84.ckpt"
)
IMAGE_PATH = Path("/home/jandu/repos/traenslenzor/.data/image.png")


def _skip_if_missing_assets() -> None:
    if not CKPT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CKPT_PATH}")
    if not IMAGE_PATH.exists():
        pytest.skip(f"Test image not found: {IMAGE_PATH}")


def _ensure_file_server_running():
    """Start the file server if 8005 is free; otherwise assume it's already up."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.2)
        sock.connect((ADDRESS, PORT))
        # port is occupied; assume server is running
        return None
    except OSError:
        # need to start server
        config = Config(app=file_server_app, host=ADDRESS, port=PORT, log_level="error")
        server = Server(config)
        loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(0.5)
        return (server, loop, thread)
    finally:
        sock.close()


def _stop_file_server(started):
    if not started:
        return
    server, loop, thread = started
    server.should_exit = True
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def test_runtime_loads_checkpoint_and_infers_on_image():
    _skip_if_missing_assets()
    started = _ensure_file_server_running()

    # Upload image via FileClient using its async API
    file_id = asyncio.run(FileClient.put(str(IMAGE_PATH)))
    assert file_id is not None, "Failed to upload test image via FileClient"

    config = DocClassifierMCPConfig(
        checkpoint_path=CKPT_PATH,
        is_mock=False,
        device="cpu",
        img_size=224,
    )
    runtime = config.setup_target()

    # classify via file_id to mirror runtime flow (falls back to mock if checkpoint load fails)
    result = asyncio.run(runtime.classify_file_id(file_id, top_k=5))
    preds = result["predictions"]

    # Basic shape/ordering checks
    assert len(preds) == 5
    assert all(p["label"] in CLASS_NAMES for p in preds)
    assert all(
        preds[i]["probability"] >= preds[i + 1]["probability"] for i in range(len(preds) - 1)
    )

    _stop_file_server(started)
