import subprocess
import sys
from pathlib import Path


async def run():
    app = Path(__file__).parent / "app.py"
    port = 8080

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app,
        f"--server.port={port}",
        "--server.headless=true",
    ]
    subprocess.Popen(cmd, stdout=sys.stdout)
