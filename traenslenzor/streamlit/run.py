import os
import subprocess
import sys
from pathlib import Path
from typing import Final

_DEFAULT_PORT: Final = 8080
_DEFAULT_ADDRESS: Final = "localhost"
_DEFAULT_HEADLESS: Final = True


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


async def run() -> None:
    app = Path(__file__).parent / "app.py"
    port = _env_int("STREAMLIT_SERVER_PORT", _DEFAULT_PORT)
    address = os.getenv("STREAMLIT_SERVER_ADDRESS", _DEFAULT_ADDRESS)
    headless = _env_bool("STREAMLIT_SERVER_HEADLESS", _DEFAULT_HEADLESS)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app,
        f"--server.port={port}",
        f"--server.address={address}",
        f"--server.headless={'true' if headless else 'false'}",
    ]
    subprocess.Popen(cmd, stdout=sys.stdout)
