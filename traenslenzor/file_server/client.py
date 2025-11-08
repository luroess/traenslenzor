from pathlib import Path
from typing import Optional

import aiofiles
import httpx

from traenslenzor.file_server.server import ADDRESS, PORT

FILES_ENDPOINT = f"http://{ADDRESS}:{PORT}/files"


class FileClient:
    @staticmethod
    async def put(filepath: str) -> Optional[str]:
        """Upload a file and return its UUID string, or None on failure."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        async with aiofiles.open(path, "rb") as f:
            file_data = await f.read()

        async with httpx.AsyncClient() as client:
            resp = await client.post(FILES_ENDPOINT, files={"file": (path.name, file_data)})

        return None if not resp.is_success else str(resp.json().get("id"))

    @staticmethod
    async def put_bytes(name: str, data: bytes) -> Optional[str]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                FILES_ENDPOINT,
                files={"file": (name, data)},
            )
        return None if not resp.is_success else str(resp.json().get("id"))

    @staticmethod
    async def get(file_id: str) -> Optional[bytes]:
        """Download a file and return its bytes, or None if not found."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{FILES_ENDPOINT}/{file_id}")

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        return resp.content

    @staticmethod
    async def rem(file_id: str) -> bool:
        """Remove a file; return True if deleted, False if not found."""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f"{FILES_ENDPOINT}/{file_id}")

        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True
