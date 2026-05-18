"""Simple in-process rate limiter (sliding window of timestamps)."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Allow at most ``max_calls`` invocations per ``period`` seconds.

    Thread-safe; calls block until a slot is available.
    """

    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.max_calls = max_calls
        self.period = float(period)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps outside the window.
                while self._calls and (now - self._calls[0]) >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            if wait > 0:
                time.sleep(wait)
