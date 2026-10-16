"""Keeping the pace of requests below a limit per time window.

A free plan allows only so many requests (or tokens) a minute. Waiting here, before a request is sent, is better than
sending it and being refused: a refusal still counts, and it costs a retry.
"""

import time
from collections import deque
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Throttle:
    """Lets at most ``limit`` units through in any window of ``window_seconds``, making callers wait when needed.

    A unit is a request, or a token when the limit is counted in tokens. The clock and the pause are replaceable so
    that tests do not really wait.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be above 0")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = Lock()

    def acquire(self, units: int = 1) -> float:
        """Wait until ``units`` more units fit in the window, then count them.

        A request that is larger than the whole limit goes through when the window is empty, so that it cannot wait
        for ever.

        Args:
            units: How many units the request uses.

        Returns:
            How many seconds were spent waiting.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= self._window:
                    self._events.popleft()
                used = sum(count for _, count in self._events)
                if not self._events or used + units <= self._limit:
                    self._events.append((now, units))
                    return waited
                wait = max(self._events[0][0] + self._window - now, 0.0)
            self._sleep(wait)
            waited += wait
