"""FastMCP server for image rendering and text replacement."""

from pathlib import Path

from fastmcp import FastMCP

from traenslenzor.file_server.client import FileClient
from traenslenzor.image_renderer.image_renderer import ImageRenderer
from traenslenzor.image_renderer.text_operations import Text

ADDRESS = "127.0.0.1"
PORT = 8002
IMAGE_RENDERER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

# Initialize FastMCP server
image_renderer_mcp = FastMCP("Image Renderer")

# Singleton instance for model persistence
_renderer_instance: ImageRenderer | None = None


def get_renderer(device: str = "mps") -> ImageRenderer:
    """Get or create the singleton ImageRenderer instance."""
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = ImageRenderer(device=device)
    return _renderer_instance


@image_renderer_mcp.tool
async def replace_text(
    image_id: str,
    texts: list[Text],
) -> str:
    """
    Replace text in an image using inpainting.

    Args:
        image_id: ID of the image file (from FileClient)
        texts: List of text regions to replace with new content
    Returns:
        ID of the result image
    """

    # Debug Config
    debug_dir = "./debug"
    save_debug = True

    # Load image from FileClient
    image = await FileClient.get_image(image_id)
    assert image is not None, f"Image with ID '{image_id}' not found"

    # Process image
    renderer = get_renderer(device="mps")
    result_image = await renderer.replace_text(
        image=image,
        texts=texts,
        inverse_transformation=None,
        save_debug=save_debug,
        debug_dir=debug_dir,
    )

    if save_debug:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        result_image.save(debug_path / "debug-result.png")

    result_id = await FileClient.put_img(f"rendered_{image_id}", result_image)
    assert result_id is not None, "Failed to save result image"

    return result_id


async def run():
    """Run the FastMCP server."""
    await image_renderer_mcp.run_async(transport="streamable-http", port=PORT, host=ADDRESS)
