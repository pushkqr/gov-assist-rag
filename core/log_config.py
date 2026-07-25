import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured via the DEBUG environment variable.

    - DEBUG=true  → DEBUG level (verbose per-chunk progress)
    - DEBUG=false → INFO level  (pipeline milestones only)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        debug_on = os.getenv("DEBUG", "false").strip().lower() in ("true", "1", "yes")
        level = logging.DEBUG if debug_on else logging.INFO
        handler.setLevel(level)
        logger.setLevel(level)

        fmt = logging.Formatter("[%(name)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
