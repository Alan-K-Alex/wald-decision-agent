from __future__ import annotations

import logging
from pathlib import Path

from .config import AppSettings


LOGGER_NAME = "wald_decision_agent"


def configure_logging(settings: AppSettings) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(settings.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
