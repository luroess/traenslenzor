import logging
import os

# disable PADDLEOCR logger scheme
os.environ["PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG"] = "1"

from traenslenzor.app import app
from traenslenzor.logging_config import setup_logger


def main():
    setup_logger()
    app.run()


if __name__ == "__main__":
    main()
