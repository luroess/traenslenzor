from io import BytesIO
from pathlib import Path
from typing import Optional

import aiofiles
import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

from traenslenzor.file_server.server import ADDRESS, PORT
from traenslenzor.image_utils.image_utils import pil_to_numpy

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
    async def put_img(img_name: str, img: PILImage) -> Optional[str]:
        """Upload a file and return its UUID string, or None on failure."""

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        async with httpx.AsyncClient() as client:
            resp = await client.post(FILES_ENDPOINT, files={"file": (img_name, buffer.getvalue())})

        if not resp.is_success:
            return None

        return str(resp.json().get("id"))

    @staticmethod
    async def get_raw_bytes(file_id: str) -> Optional[bytes]:
        """Download a file and return its bytes, or None if not found."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{FILES_ENDPOINT}/{file_id}")

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        return resp.content

    @staticmethod
    async def get_image(file_id: str) -> PILImage | None:
        """Download a image and return as a PILImage, or None if not found."""
        file_bytes = await FileClient.get_raw_bytes(file_id)

        if file_bytes is None:
            return None

        return Image.open(BytesIO(file_bytes))

    @staticmethod
    async def get_image_as_numpy(file_id: str) -> NDArray[np.float32] | None:
        """Download a image and return as a PILImage, or None if not found."""
        img = await FileClient.get_image(file_id)

        if img is None:
            return None

        return pil_to_numpy(img)

    @staticmethod
    async def rem(file_id: str) -> bool:
        """Remove a file; return True if deleted, False if not found."""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f"{FILES_ENDPOINT}/{file_id}")

        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True
