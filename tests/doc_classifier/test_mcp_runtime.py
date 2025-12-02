from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traenslenzor.doc_classifier.configs import DocClassifierMCPConfig
from traenslenzor.doc_classifier.mcp.runtime import CLASS_NAMES
from traenslenzor.file_server.client import FileClient


def _make_image(path: Path, size: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
    return path


def test_classify_file_id_mock_is_deterministic(running_file_server, tmp_path):
    img_path = _make_image(tmp_path / "doc.png")
    file_id = asyncio.run(FileClient.put(str(img_path)))
    assert file_id is not None

    runtime = DocClassifierMCPConfig(img_size=64, is_mock=True, checkpoint_path=None).setup_target()

    out1 = asyncio.run(runtime.classify_file_id(file_id, top_k=3))["predictions"]
    out2 = asyncio.run(runtime.classify_file_id(file_id, top_k=3))["predictions"]

    assert out1 == out2  # deterministic hash-based mock
    assert len(out1) == 3
    assert all(p["label"] in CLASS_NAMES for p in out1)

    probs = [p["probability"] for p in out1]
    assert probs == pytest.approx(sorted(probs, reverse=True))


def test_classify_path_respects_top_k(tmp_path):
    img_path = _make_image(tmp_path / "doc2.png")
    runtime = DocClassifierMCPConfig(is_mock=True, checkpoint_path=None).setup_target()

    file_id = asyncio.run(FileClient.put(str(img_path)))
    assert file_id is not None

    result = asyncio.run(runtime.classify_file_id(file_id, top_k=5))
    preds = result["predictions"]

    assert len(preds) == 5
    # Returned labels are unique top indices and sorted by probability
    assert all(preds[i]["probability"] >= preds[i + 1]["probability"] for i in range(4))
