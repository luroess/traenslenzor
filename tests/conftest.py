"""Pytest configuration for traenslenzor tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def image_diff_reference_dir() -> Path:
    """Store image regression snapshots in tests/snapshots directory."""
    return Path(__file__).parent / "snapshots"
