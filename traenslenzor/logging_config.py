"""Logging configuration for traenslenzor."""
import logging
import os


def setup_logger():
    """Setup logging configuration for the application."""
    lvl = str(os.getenv("TRAENSLENZOR_LOG_LEVEL", "INFO")).upper()
    level = logging.getLevelNamesMapping().get(lvl, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)

    for handler in root.handlers:
        handler.setLevel(level)
