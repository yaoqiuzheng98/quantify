"""Loguru-based logger setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from quantify.config import get_settings

_CONFIGURED = False


def setup_logger() -> "logger":
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    settings = get_settings()
    log_dir = Path(settings.log.dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        log_dir / "quantify_{time:YYYYMMDD}.log",
        rotation="00:00",
        retention="30 days",
        level=settings.log.level,
        encoding="utf-8",
        enqueue=True,
    )
    _CONFIGURED = True
    return logger


log = setup_logger()
