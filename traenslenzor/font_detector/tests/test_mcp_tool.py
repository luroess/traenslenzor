import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from traenslenzor.file_server.session_state import BBoxPoint, SessionState, TextItem
from traenslenzor.font_detector.server import detect_font_logic


def test_detect_font_mcp_tool():
    """
    Test the detect_font MCP tool by mocking external dependencies.
    This verifies the orchestration logic without needing a running File Server.
    """

    async def run_test():
        # 1. Setup Mock Data
        session_id = "test-session-123"
        fake_image_id = "fake-image-id"

        # Create a fake session state with one text item
        # BBox is 100x20 pixels
        text_item = TextItem(
            extractedText="Hello World",
            confidence=0.99,
            bbox=[
                BBoxPoint(x=10, y=10),  # UL
                BBoxPoint(x=110, y=10),  # UR
                BBoxPoint(x=110, y=30),  # LR
                BBoxPoint(x=10, y=30),  # LL
            ],
        )
        session = SessionState(rawDocumentId=fake_image_id, text=[text_item])

        # 2. Mock Dependencies
        with (
            patch("traenslenzor.font_detector.server.SessionClient") as MockSessionClient,
            patch("traenslenzor.font_detector.server.FileClient") as MockFileClient,
            patch(
                "traenslenzor.font_detector.server.get_font_name_detector"
            ) as mock_get_name_detector,
            patch(
                "traenslenzor.font_detector.server.get_font_size_estimator"
            ) as mock_get_size_estimator,
        ):
            # Mock SessionClient.get to return our fake session
            MockSessionClient.get = AsyncMock(return_value=session)

            # Mock SessionClient.update to apply the update function immediately
            async def fake_update(sid, update_func):
                update_func(session)

            MockSessionClient.update = AsyncMock(side_effect=fake_update)

            # Mock FileClient to return fake image bytes
            MockFileClient.get_raw_bytes = AsyncMock(return_value=b"fake_png_bytes")

            # Mock FontNameDetector
            mock_name_detector = MagicMock()
            mock_name_detector.detect.return_value = "Arial-Regular"
            mock_get_name_detector.return_value = mock_name_detector

            # Mock FontSizeEstimator
            mock_size_estimator = MagicMock()
            mock_size_estimator.estimate.return_value = 16.0
            mock_get_size_estimator.return_value = mock_size_estimator

            # 3. Run the Tool
            result = await detect_font_logic(session_id)

            # 4. Verify Results

            # Check return message
            assert "successful" in result
            assert "Arial-Regular" in result

            # Check that the session object was actually updated
            assert session.text[0].detectedFont == "Arial-Regular"
            assert session.text[0].font_size == 16

            # Check that the correct calls were made
            MockSessionClient.get.assert_called_with(session_id)
            MockFileClient.get_raw_bytes.assert_called_with(fake_image_id)

            # Verify size estimator was called with correct dimensions
            # Width should be 100, Height 20
            args, kwargs = mock_size_estimator.estimate.call_args
            text_box_size = kwargs.get("text_box_size")
            assert text_box_size is not None
            assert abs(text_box_size[0] - 100.0) < 0.001
            assert abs(text_box_size[1] - 20.0) < 0.001

    asyncio.run(run_test())
