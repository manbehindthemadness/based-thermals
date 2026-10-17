"""
Time shutdown stages without changing resource cleanup or exception handling.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from logging import Logger
from time import perf_counter
from typing import Any


@contextmanager
def shutdown_stage(logger: Logger, diagnostics: Any, name: str) -> Iterator[None]:
    """
    Publish cleanup stages and log durations at debug level, including failed cleanup.
    """
    diagnostics.stage(f"shutdown/{name}")
    logger.debug("Shutdown: %s started", name)
    started = perf_counter()
    try:
        yield
    finally:
        logger.debug("Shutdown: %s finished in %.3f s", name, perf_counter() - started)
