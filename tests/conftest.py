"""Pytest configuration for traenslenzor tests."""

import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

from traenslenzor.file_server.client import FileClient
from traenslenzor.file_server.server import ADDRESS, PORT


@pytest.fixture
def anyio_backend():
    """Configure anyio to only use asyncio backend."""
    return "asyncio"


@pytest.fixture(scope="session")
def image_diff_reference_dir() -> Path:
    """Store image regression snapshots in tests/snapshots directory."""
    return Path(__file__).parent / "snapshots"


@pytest.fixture(scope="session")
def file_server() -> Generator[type[FileClient], None, None]:
    """Start the file server as a background process for tests."""
    import subprocess

    # Start the server in background
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "traenslenzor.file_server.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready (max 5 seconds)
    ready = False
    for _ in range(50):
        try:
            response = httpx.get(f"http://{ADDRESS}:{PORT}/docs", timeout=1.0)
            if response.status_code == 200:
                ready = True
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.1)

    if not ready:
        process.kill()
        raise RuntimeError("File server failed to start")

    yield FileClient

    # Teardown: stop the server
    process.terminate()
    process.wait(timeout=5)
