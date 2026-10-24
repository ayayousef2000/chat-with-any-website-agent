"""Waiting out rate limits and brief outages of outside services.

Free plans limit how fast requests may come. When a service says "too many requests" or is briefly unavailable, the
right response is to wait for the time it names, or for a growing pause when it names none, and to try again, within
a bound so that nobody is left waiting for minutes. When the bound is reached, a ``ServiceBusyError`` says how long to
wait. Each client says which of its errors are worth trying again; the others would fail the same way again.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chat_with_any_website_agent.errors import ServiceBusyError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


def retry_after_seconds(error: BaseException) -> float | None:
    """Read how long a service asked callers to wait, from the headers of its error response.

    Args:
        error: An error raised by a service client. The headers are looked for on its ``response`` and on the error.

    Returns:
        The wait in seconds, or ``None`` if the service named none.
    """
    headers: Mapping[str, str] | None = None
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if not headers:
        return None
    lowered = {str(key).lower(): value for key, value in headers.items()}
    try:
        if "retry-after-ms" in lowered:
            return max(float(lowered["retry-after-ms"]) / 1000, 0.0)
        if "retry-after" in lowered:
            return max(float(lowered["retry-after"]), 0.0)
    except TypeError, ValueError:
        return None
    return None


def describe_wait(seconds: float | None) -> str:
    """Say how long to wait in words, for a message to a person.

    Args:
        seconds: The wait the service asked for, or ``None`` if it named none.

    Returns:
        For example ``"about 23 seconds"``. A wait the service did not name is ``"a few seconds"`` (these limits
        are counted per minute, so a place usually frees up quickly), a long one is ``"a few minutes"``, and one of
        more than an hour and a half, such as the end of a daily limit, is counted in hours (``"about 3 hours"``).
    """
    if seconds is None:
        return "a few seconds"
    if seconds > 5400:
        hours = round(seconds / 3600)
        return f"about {hours} hours"
    if seconds > 90:
        return "a few minutes"
    rounded = max(1, round(seconds))
    return f"about {rounded} second{'s' if rounded != 1 else ''}"


@dataclass(frozen=True)
class RetryPolicy:
    """How patiently to retry.

    Attributes:
        attempts: The most tries, including the first.
        base_delay: The first pause when the service names none; it doubles each time.
        max_delay: The longest single pause. A service that asks for more is not waited for.
        budget: The longest total time spent pausing.
    """

    attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 20.0
    budget: float = 45.0


def _spread(delay: float) -> float:
    """Add up to a quarter of a pause at random, so that many callers do not all return at the same moment."""
    return random.uniform(0, delay * 0.25)  # noqa: S311 - this spreads out retries; it is not a security use


@dataclass
class Retrier:
    """Runs an operation again after a transient failure, as its policy allows.

    The pause and the randomness are replaceable so that tests do not really wait.

    Attributes:
        retryable: The errors that are worth trying again.
        policy: How patiently to retry.
        sleep: How to pause.
        jitter: How much to add to a pause, given the pause.
    """

    retryable: tuple[type[BaseException], ...]
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[float], float] = field(default=lambda delay: _spread(delay))

    def call[T](self, operation: Callable[[], T], *, name: str = "The service") -> T:
        """Run ``operation``, trying again when a retryable error occurs.

        Args:
            operation: The call to make. It should do one request.
            name: What the call is, for the log and for the message to the person.

        Returns:
            What ``operation`` returns.

        Raises:
            ServiceBusyError: When a retryable error keeps happening until the attempts or the time budget run out,
                or when the service asks for a longer wait than allowed.
            Exception: At once, any error that retrying cannot fix.
        """
        waited = 0.0
        for attempt in range(1, self.policy.attempts + 1):
            try:
                return operation()
            except self.retryable as error:
                asked = retry_after_seconds(error)
                if attempt == self.policy.attempts or (asked is not None and asked > self.policy.max_delay):
                    raise _busy(name, asked) from error
                pause = (
                    asked
                    if asked is not None
                    else min(self.policy.base_delay * 2 ** (attempt - 1), self.policy.max_delay)
                )
                pause += self.jitter(pause)
                if waited + pause > self.policy.budget:
                    raise _busy(name, asked if asked is not None else pause) from error
                logger.warning(
                    "%s failed with %s; trying again in %.1f s (attempt %d of %d)",
                    name,
                    type(error).__name__,
                    pause,
                    attempt + 1,
                    self.policy.attempts,
                )
                self.sleep(pause)
                waited += pause
        raise AssertionError("unreachable")  # pragma: no cover - the loop always returns or raises


def _busy(name: str, wait: float | None) -> ServiceBusyError:
    message = f"{name} is busy right now. Try again in {describe_wait(wait)}."
    return ServiceBusyError(message, retry_after=wait, service=name)
