from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_attempts() -> int:
    return max(1, int(os.getenv("RESEARCH_RETRY_ATTEMPTS", "3")))


def retry_base_delay() -> float:
    return max(0.0, float(os.getenv("RESEARCH_RETRY_BASE_SECONDS", "1")))


def call_with_retries(
    operation: Callable[[], T],
    is_retryable: Callable[[Exception], bool],
    attempts: int | None = None,
    base_delay: float | None = None,
) -> T:
    total_attempts = attempts if attempts is not None else retry_attempts()
    delay = base_delay if base_delay is not None else retry_base_delay()
    for attempt in range(1, total_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= total_attempts or not is_retryable(exc):
                raise
            time.sleep(delay * (2 ** (attempt - 1)))
    raise RuntimeError("retry loop exited unexpectedly")
