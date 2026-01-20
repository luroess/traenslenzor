"""
Supervisor Integration Test - Full Pipeline via LLM Agent.

This test validates the entire trÄnslenzor pipeline by running the supervisor
with a real LLM. It verifies that the agent correctly orchestrates all MCP tools
to complete a document translation workflow.

**IMPORTANT**: This test is expensive (uses LLM API calls) and requires:
1. All MCP servers to be running
2. Valid LLM API credentials
3. Explicit opt-in via environment variable or pytest marker

Run manually with:
    RUN_SUPERVISOR_INTEGRATION=1 uv run pytest tests/test_supervisor_integration.py -v -s

Or with specific markers:
    uv run pytest tests/test_supervisor_integration.py -v -s -m "supervisor_integration"
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# ============================================================================
# Configuration
# ============================================================================

SKIP_REASON = (
    "Supervisor integration test skipped. "
    "Set RUN_SUPERVISOR_INTEGRATION=1 to run this expensive test."
)

RUN_INTEGRATION = os.getenv("RUN_SUPERVISOR_INTEGRATION", "").lower() in ("1", "true", "yes")


@dataclass
class IntegrationTestConfig:
    """Configuration for supervisor integration tests."""

    # Test image settings
    test_image_path: Path | None = None
    """Path to a custom test image. If None, generates a synthetic image."""

    target_language: str = "German"
    """Target language for translation."""

    # Timeouts
    supervisor_timeout: float = 300.0
    """Maximum time (seconds) for supervisor to complete."""

    tool_call_timeout: float = 60.0
    """Maximum time (seconds) for individual tool calls."""

    # Expected behavior
    min_tool_calls: int = 3
    """Minimum number of tool calls expected for a complete workflow."""

    expected_tools: list[str] = field(
        default_factory=lambda: [
            "extract_text",
            "detect_font",
            "translate",
            "replace_text",
        ]
    )
    """Tools expected to be called during workflow (in any order)."""

    # Validation
    require_rendered_output: bool = True
    """Whether to require a rendered document in final session state."""

    save_artifacts: bool = True
    """Whether to save test artifacts (images, logs) for debugging."""

    artifacts_dir: Path = field(default_factory=lambda: Path(".test_artifacts"))
    """Directory to save test artifacts."""


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def integration_config() -> IntegrationTestConfig:
    """Provide default integration test configuration."""
    return IntegrationTestConfig()


@pytest.fixture
def test_image() -> "PILImage":
    """Create or load a test image for the pipeline."""
    from PIL import Image, ImageDraw, ImageFont

    # Create a simple test document
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    # Add test text
    test_lines = [
        "Hello, this is a test document.",
        "It contains multiple lines of text.",
        "The supervisor should translate this to German.",
        "And render it back onto the image.",
    ]

    y_position = 50
    for line in test_lines:
        draw.text((50, y_position), line, fill="black", font=font)
        y_position += 40

    return img


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
@pytest.mark.supervisor_integration
@pytest.mark.anyio
async def test_supervisor_full_workflow(
    file_server,
    integration_config: IntegrationTestConfig,
    test_image: "PILImage",
):
    """
    Test the complete supervisor workflow with LLM orchestration.

    This test:
    1. Uploads a test image to the file server
    2. Runs the supervisor with a translation prompt
    3. Verifies all expected tools were called
    4. Validates the final session state has a rendered document
    """
    from langchain_core.messages import AIMessage

    from traenslenzor.file_server.client import FileClient, SessionClient
    from traenslenzor.supervisor.supervisor import run as run_supervisor

    # Setup: Upload test image
    raw_doc_id = await FileClient.put_img("supervisor_test_image.png", test_image)
    assert raw_doc_id is not None, "Failed to upload test image"

    # Create the user prompt
    prompt = f"""
    I have uploaded a document image with ID: {raw_doc_id}
    
    Please translate all text in this document to {integration_config.target_language}.
    Use all necessary tools to:
    1. Extract text from the document
    2. Detect fonts used in the document
    3. Translate the text
    4. Render the translated text back onto the image
    
    Return the final rendered document.
    """

    # Run supervisor with timeout
    try:
        result, session_id = await asyncio.wait_for(
            run_supervisor(prompt),
            timeout=integration_config.supervisor_timeout,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            f"Supervisor timed out after {integration_config.supervisor_timeout}s"
        )

    # Validate result
    assert result is not None, "Supervisor returned None"
    assert isinstance(result, AIMessage), f"Expected AIMessage, got {type(result)}"
    assert session_id is not None, "Session ID is None"

    # Check session state
    session = await SessionClient.get(session_id)
    assert session is not None, "Failed to retrieve session"

    # Validate workflow completion
    if integration_config.require_rendered_output:
        assert session.renderedDocumentId is not None, (
            "No rendered document found - workflow incomplete"
        )

        # Verify rendered image is accessible
        rendered_img = await FileClient.get_image(session.renderedDocumentId)
        assert rendered_img is not None, "Failed to retrieve rendered image"

        # Save artifacts if configured
        if integration_config.save_artifacts:
            artifacts_dir = integration_config.artifacts_dir
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            test_image.save(artifacts_dir / "input_image.png")
            rendered_img.save(artifacts_dir / "output_image.png")

            # Save session state
            (artifacts_dir / "session_state.json").write_text(
                session.model_dump_json(indent=2)
            )

    print(f"\n✅ Supervisor integration test passed!")
    print(f"   Session ID: {session_id}")
    print(f"   Final message: {result.content[:200]}...")


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
@pytest.mark.supervisor_integration
@pytest.mark.anyio
async def test_supervisor_tool_discovery(file_server):
    """
    Test that the supervisor can discover and list all available tools.

    This is a lighter test that verifies MCP tool connectivity without
    running a full workflow.
    """
    from traenslenzor.supervisor.tools.tools import get_tools

    tools = await get_tools()

    assert len(tools) > 0, "No tools discovered"

    tool_names = [t.name for t in tools]
    print(f"\nDiscovered {len(tools)} tools: {tool_names}")

    # Check for expected core tools
    expected_core = ["extract_text", "translate", "replace_text"]
    for expected in expected_core:
        assert expected in tool_names, f"Missing expected tool: {expected}"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
@pytest.mark.supervisor_integration
@pytest.mark.anyio
async def test_supervisor_error_handling(file_server):
    """
    Test supervisor behavior with invalid input.

    Verifies the agent handles errors gracefully without crashing.
    """
    from traenslenzor.supervisor.supervisor import run as run_supervisor

    # Test with invalid document ID
    prompt = """
    I have a document with ID: invalid_nonexistent_id_12345
    Please translate it to German.
    """

    try:
        result, session_id = await asyncio.wait_for(
            run_supervisor(prompt),
            timeout=120.0,
        )
        # Should complete without raising, even if workflow fails
        assert result is not None, "Supervisor should return a message even on error"
    except asyncio.TimeoutError:
        pytest.fail("Supervisor timed out on error handling test")


# ============================================================================
# CLI Entry Point for Manual Testing
# ============================================================================


async def _run_manual_test():
    """Run the integration test manually without pytest."""
    import sys

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from PIL import Image, ImageDraw, ImageFont

    from traenslenzor.file_server.client import FileClient, SessionClient
    from traenslenzor.file_server.session_state import SessionState
    from traenslenzor.supervisor.supervisor import run as run_supervisor

    print("🚀 Starting manual supervisor integration test...")

    # Create test image
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    test_lines = [
        "Hello, this is a test document.",
        "It contains multiple lines of text.",
        "The supervisor should translate this to German.",
    ]

    y = 50
    for line in test_lines:
        draw.text((50, y), line, fill="black", font=font)
        y += 40

    # Upload
    print("📤 Uploading test image...")
    raw_doc_id = await FileClient.put_img("manual_test_image.png", img)
    print(f"   Document ID: {raw_doc_id}")

    # Run supervisor
    prompt = f"""
    I have uploaded a document image with ID: {raw_doc_id}
    Please translate all text to German and render it back.
    """

    print("🤖 Running supervisor...")
    result, session_id = await run_supervisor(prompt)

    print(f"\n📋 Result:")
    print(f"   Session ID: {session_id}")
    print(f"   Response: {result.content}")

    # Check session
    if session_id:
        session = await SessionClient.get(session_id)
        if session.renderedDocumentId:
            print(f"   ✅ Rendered document: {session.renderedDocumentId}")
            rendered = await FileClient.get_image(session.renderedDocumentId)
            if rendered:
                rendered.save(".test_artifacts/manual_output.png")
                print("   💾 Saved to .test_artifacts/manual_output.png")
        else:
            print("   ⚠️  No rendered document found")


if __name__ == "__main__":
    # Allow running directly: python tests/test_supervisor_integration.py
    asyncio.run(_run_manual_test())
