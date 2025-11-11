"""MCP server for font detection and size estimation."""

from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .font_name_detector import FontNameDetector
from .font_size_model.infer import FontSizeEstimator

# Initialize server
app = Server("font-detector")

# Global instances (lazy loaded)
font_name_detector_instance: Optional[FontNameDetector] = None
font_size_estimator_instance: Optional[FontSizeEstimator] = None


def get_font_name_detector() -> FontNameDetector:
    """Get or create font name detector instance."""
    global font_name_detector_instance
    if font_name_detector_instance is None:
        font_name_detector_instance = FontNameDetector()
    return font_name_detector_instance


def get_font_size_estimator() -> FontSizeEstimator:
    """Get or create font size estimator instance."""
    global font_size_estimator_instance
    if font_size_estimator_instance is None:
        # Use checkpoints in the font_detector directory
        checkpoints_dir = Path(__file__).parent / "checkpoints"
        font_size_estimator_instance = FontSizeEstimator(str(checkpoints_dir))
    return font_size_estimator_instance


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="detect_font_name",
            description="Detect font name from an image containing text",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file containing text",
                    },
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="estimate_font_size",
            description="Estimate font size in points from text box dimensions and content",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_box_size": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Text box dimensions as [width_px, height_px]",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content in the box",
                    },
                    "font_name": {
                        "type": "string",
                        "description": "Optional font name hint (if known)",
                    },
                },
                "required": ["text_box_size", "text"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "detect_font_name":
        # Validate input
        image_path = arguments.get("image_path")
        if not image_path:
            return [
                TextContent(
                    type="text",
                    text='{"error": "image_path is required"}',
                )
            ]

        try:
            # Detect font name
            detector = get_font_name_detector()
            font_name = detector.detect(image_path)

            # Return result
            import json

            result = {"font_name": font_name}
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result),
                )
            ]

        except Exception as e:
            import json

            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}),
                )
            ]

    elif name == "estimate_font_size":
        # Validate input
        text_box_size = arguments.get("text_box_size")
        text = arguments.get("text")
        font_name_raw: Any | None = arguments.get("font_name")
        font_param: str = str(font_name_raw) if font_name_raw else ""

        if not text_box_size or not isinstance(text_box_size, list) or len(text_box_size) != 2:
            import json

            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "text_box_size must be a 2-element array"}),
                )
            ]

        if not text:
            import json

            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "text is required"}),
                )
            ]

        try:
            # If no font name provided, detect it
            if not font_param:
                # For now, we require font_name to be provided
                # In a full implementation, we would render the text and detect
                import json

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "font_name is required (automatic detection from text not yet implemented)"
                            }
                        ),
                    )
                ]

            # Estimate font size
            estimator = get_font_size_estimator()
            font_size_pt = estimator.estimate(
                text_box_size=tuple(text_box_size),
                text=text,
                font_name=font_param,
            )

            # Return result
            import json

            size_result: dict[str, Any] = {"font_size_pt": font_size_pt}
            return [
                TextContent(
                    type="text",
                    text=json.dumps(size_result),
                )
            ]

        except Exception as e:
            import json

            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}),
                )
            ]

    else:
        import json

        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )
        ]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
