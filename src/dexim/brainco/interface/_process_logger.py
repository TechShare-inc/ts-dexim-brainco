"""Process logger configuration for BrainCo interface child processes."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_BANNER_WIDTH = 70
_BANNER_SEP = "=" * _BANNER_WIDTH


def _log_process_banner(process_name: str | None, log_file: str | None) -> None:
    from loguru import logger

    pid = os.getpid()
    py_ver = platform.python_version()
    plat = platform.platform(aliased=True, terse=True)
    log_label = log_file if log_file is not None else "<stderr only>"

    name_part = f"  -  {process_name}" if process_name else ""
    logger.info(_BANNER_SEP)
    logger.info(f"  DexImitate{name_part}  -  PID {pid}")
    logger.info(f"  Python {py_ver}  -  {plat}")
    logger.info(f"  Log: {log_label}")
    logger.info(_BANNER_SEP)


def configure_process_logger(
    log_file: str | None,
    process_name: str | None = None,
) -> None:
    """Reconfigure loguru for a child process.

    Each spawned process starts with a fresh Python interpreter (``spawn``
    start method on Windows/macOS), so loguru's sink configuration from the
    parent is *not* inherited.  Call this once at the very top of every
    process entry-point function to restore sensible logging behaviour.

    A stderr sink is always installed (INFO+).  When *log_file* is given a
    rotating file sink is added (TRACE+) with ``enqueue=True`` so file I/O
    runs on a background thread and never blocks the real-time control loop.

    Args:
        log_file: Absolute or relative path for the process log file.
            ``None`` disables file logging (stderr only).
        process_name: Short human-readable label for this process
            (e.g. ``"hw-core"``).  Included in the startup banner.
            ``None`` omits the process-name field.
    """
    from loguru import logger

    # Remove any pre-existing sinks (fresh logger in spawn context).
    logger.remove()

    # Ensure the log directory exists.
    if log_file is not None:
        log_dir = Path(log_file).parent
        if log_dir != Path("."):
            log_dir.mkdir(parents=True, exist_ok=True)

    # Stderr sink: human-readable, INFO+.
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        enqueue=True,
    )

    # File sink: TRACE+ with rotation.
    if log_file is not None:
        logger.add(
            str(log_file),
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSSSSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level="TRACE",
            rotation="10 MB",
            retention=3,
            enqueue=True,
        )

    _log_process_banner(process_name, log_file)
