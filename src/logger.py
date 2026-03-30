import logging
import os
import sys
from src.config import LOG_PATH


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs
    if logger.handlers:
        return logger

    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # -----------------------------
    # FILE HANDLER (logs to file)
    # -----------------------------
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # -----------------------------
    # CONSOLE HANDLER (logs to terminal)
    # -----------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # -----------------------------
    # FORMATTER
    # -----------------------------
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # -----------------------------
    # ADD HANDLERS
    # -----------------------------
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger