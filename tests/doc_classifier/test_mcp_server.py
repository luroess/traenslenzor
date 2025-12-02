from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from traenslenzor.doc_classifier.mcp.runtime import CLASS_NAMES
from traenslenzor.doc_classifier.mcp.schemas import DocClassifierRequest
from traenslenzor.file_server.client import FileClient


def _make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
    return path


def test_classify_document_returns_probability_map(running_file_server, tmp_path):
    # Reload the server module to reset the cached runtime singleton
    mcp_server = importlib.import_module("traenslenzor.doc_classifier.mcp.mcp_server")
    importlib.reload(mcp_server)

    img_path = _make_image(tmp_path / "doc.png")
    file_id = asyncio.run(FileClient.put(str(img_path)))
    assert file_id is not None

    response = asyncio.run(mcp_server.classify_document(document_id=file_id, top_k=2))

    assert len(response.probabilities) == 2
    assert set(response.probabilities.keys()).issubset(set(CLASS_NAMES))


def test_doc_classifier_request_validates_top_k():
    with pytest.raises(ValidationError):
        DocClassifierRequest(document_id="abc123", top_k=0)


def test_runtime_global_cache_can_be_reset(running_file_server, tmp_path):
    mcp_server = importlib.import_module("traenslenzor.doc_classifier.mcp.mcp_server")
    importlib.reload(mcp_server)

    img_path = _make_image(tmp_path / "doc2.png")
    file_id = asyncio.run(FileClient.put(str(img_path)))
    assert file_id is not None

    # First call populates the singleton
    asyncio.run(mcp_server.classify_document(document_id=file_id, top_k=1))
    assert mcp_server._runtime is not None

    # Reset and ensure a new runtime is created on next call
    mcp_server._runtime = None
    asyncio.run(mcp_server.classify_document(document_id=file_id, top_k=1))
    assert mcp_server._runtime is not None
