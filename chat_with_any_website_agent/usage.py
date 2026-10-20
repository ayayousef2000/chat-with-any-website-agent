"""Counting the calls and tokens a run uses, and stopping it at a limit.

Free plans have monthly and daily quotas, so a run that calls an outside service can be given a budget. Every attempt
is counted, including one that is tried again, because the service counts it too.
"""

from collections import Counter
from threading import Lock
from typing import TYPE_CHECKING

from chat_with_any_website_agent.errors import BudgetExceededError

if TYPE_CHECKING:
    from collections.abc import Mapping

COHERE = "cohere"
GROQ = "groq"


class UsageCounter:
    """Counts calls and tokens per service and raises ``BudgetExceededError`` when a limit is reached."""

    def __init__(
        self, *, max_calls: Mapping[str, int] | None = None, max_tokens: Mapping[str, int] | None = None
    ) -> None:
        """Create a counter.

        Args:
            max_calls: The most calls allowed per service. A service that is not listed has no limit.
            max_tokens: The most tokens allowed per service. A service that is not listed has no limit.
        """
        self._max_calls = dict(max_calls or {})
        self._max_tokens = dict(max_tokens or {})
        self._calls: Counter[str] = Counter()
        self._tokens: Counter[str] = Counter()
        self._lock = Lock()

    @property
    def calls(self) -> dict[str, int]:
        """The calls counted so far, per service."""
        with self._lock:
            return dict(self._calls)

    @property
    def tokens(self) -> dict[str, int]:
        """The tokens counted so far, per service."""
        with self._lock:
            return dict(self._tokens)

    def record_call(self, service: str) -> None:
        """Count one call, unless the limit of calls for the service has been reached.

        Args:
            service: The service that is about to be called.

        Raises:
            BudgetExceededError: If the service has already had as many calls as it is allowed. Nothing is counted.
        """
        with self._lock:
            limit = self._max_calls.get(service)
            if limit is not None and self._calls[service] >= limit:
                raise BudgetExceededError(f"The limit of {limit} calls to {service} has been reached.", service, limit)
            self._calls[service] += 1

    def record_tokens(self, service: str, tokens: int) -> None:
        """Count the tokens a call used, which are only known once the call is done.

        Args:
            service: The service that was called.
            tokens: The tokens the call used.

        Raises:
            BudgetExceededError: If the tokens used so far pass the limit for the service. They are counted first, so
                the totals stay true.
        """
        with self._lock:
            self._tokens[service] += tokens
            limit = self._max_tokens.get(service)
            if limit is not None and self._tokens[service] > limit:
                raise BudgetExceededError(
                    f"The limit of {limit} tokens for {service} has been reached.", service, limit
                )

    def ensure_tokens_left(self, service: str) -> None:
        """Refuse a call when the tokens of the service are already used up.

        Args:
            service: The service that is about to be called.

        Raises:
            BudgetExceededError: If the service has no tokens left.
        """
        with self._lock:
            limit = self._max_tokens.get(service)
            if limit is not None and self._tokens[service] >= limit:
                raise BudgetExceededError(
                    f"The limit of {limit} tokens for {service} has been reached.", service, limit
                )

    def tokens_left(self, service: str) -> int | None:
        """How many tokens a service may still use, or ``None`` when it has no limit."""
        with self._lock:
            limit = self._max_tokens.get(service)
            return None if limit is None else max(limit - self._tokens[service], 0)
