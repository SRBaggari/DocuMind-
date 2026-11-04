"""Configure rotating local log files. Logs must never contain document text."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "documind"


def configure_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Attach the file handler once; repeat calls, as on Streamlit reruns, change nothing."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        logs_dir / "documind.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
