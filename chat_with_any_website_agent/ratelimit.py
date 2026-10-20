"""Limits on how often one visitor may use the web server, so that nobody can use up the free plans of the services.

The limits are counted per client (the visitor's network address) over a sliding window and are kept in memory, which is
enough for a single server process. A client that is over a limit is told how long to wait.
"""

import ipaddress
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chat_with_any_website_agent.errors import TooManyRequestsError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from starlette.requests import Request

    from chat_with_any_website_agent.config import Settings

# The most clients remembered at once; when there are more, those not seen for the longest time are forgotten.
MAX_CLIENTS = 10_000


@dataclass(frozen=True)
class Rule:
    """At most ``limit`` uses in any period of ``window`` seconds.

    Attributes:
        limit: How many uses are allowed.
        window: The length of the period, in seconds.
        message: What to tell the client when the rule refuses a use. ``{wait}`` is replaced by the wait in words.
    """

    limit: int
    window: float
    message: str


@dataclass(frozen=True)
class Refusal:
    """A use that was refused: how long to wait, and the rule that refused it.

    Attributes:
        wait: How many seconds until the use would be allowed.
        rule: The rule that refused it.
    """

    wait: float
    rule: Rule


class RateLimiter:
    """Counts uses per client and action over sliding windows. Safe to use from several threads."""

    def __init__(self, rules: Mapping[str, Sequence[Rule]], clock: Callable[[], float] = time.monotonic) -> None:
        """Create a limiter.

        Args:
            rules: The rules of each action.
            clock: The time in seconds; tests replace it.
        """
        self._rules = {action: tuple(action_rules) for action, action_rules in rules.items()}
        self._clock = clock
        self._lock = threading.Lock()
        self._uses: dict[tuple[str, str], deque[float]] = {}

    def check(self, action: str, client: str) -> Refusal | None:
        """Record one use by a client, unless a rule refuses it.

        Args:
            action: The kind of use, one of the actions the limiter was created with.
            client: Who is asking.

        Returns:
            ``None`` if the use is allowed (and has been counted), otherwise the refusal with the longest wait. A
            refused use is not counted.
        """
        rules = self._rules.get(action, ())
        if not rules:
            return None
        now = self._clock()
        longest = max(rule.window for rule in rules)
        with self._lock:
            uses = self._uses.setdefault((action, client), deque())
            while uses and uses[0] <= now - longest:
                uses.popleft()

            refusal: Refusal | None = None
            for rule in rules:
                inside = [moment for moment in uses if moment > now - rule.window]
                if len(inside) >= rule.limit:
                    wait = inside[-rule.limit] + rule.window - now
                    if refusal is None or wait > refusal.wait:
                        refusal = Refusal(wait=wait, rule=rule)
            if refusal is None:
                uses.append(now)
                if len(self._uses) > MAX_CLIENTS:
                    self._forget_old_clients()
            return refusal

    def _forget_old_clients(self) -> None:
        """Drop the least recently seen clients when there are more than ``MAX_CLIENTS``.

        Every client in the table has at least one use, because a use is recorded as soon as the client is added.
        """
        excess = len(self._uses) - MAX_CLIENTS
        for key in sorted(self._uses, key=lambda key: self._uses[key][-1])[:excess]:
            del self._uses[key]


def describe_wait(seconds: float) -> str:
    """Say how long to wait in words.

    Args:
        seconds: The wait.

    Returns:
        For example ``"about 20 seconds"``, ``"about 5 minutes"`` or ``"about 3 hours"``.
    """
    seconds = max(seconds, 1.0)
    if seconds <= 90:
        count, unit = round(seconds), "second"
    elif seconds <= 5400:
        count, unit = round(seconds / 60), "minute"
    else:
        count, unit = round(seconds / 3600), "hour"
    return f"about {count} {unit}{'s' if count != 1 else ''}"


def client_id(request: Request) -> str:
    """Work out who is calling.

    This is the address of the connection. Behind a proxy every connection comes from the proxy, so the server must be
    started with uvicorn's proxy options (``--proxy-headers`` and ``--forwarded-allow-ips``, which uvicorn also reads
    from ``FORWARDED_ALLOW_IPS``), which replace the address with the visitor's own. The ``X-Forwarded-For`` header is
    deliberately not read here, because anyone can send one. IPv6 addresses are grouped by their /64 network, because
    one visitor usually controls a whole /64.

    Args:
        request: The incoming request.

    Returns:
        A text that is the same for the same visitor.
    """
    host = request.client.host if request.client else "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


class ClientLimits:
    """The limits of the server: how many questions and page loads one client may make."""

    def __init__(
        self,
        *,
        asks_per_minute: int,
        asks_per_day: int,
        loads_per_hour: int,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the limits.

        Args:
            asks_per_minute: The questions one client may ask in a minute.
            asks_per_day: The questions one client may ask in a day (a sliding 24 hours).
            loads_per_hour: The page loads and deletes one client may make in an hour.
            enabled: ``False`` turns every limit off.
            clock: The time in seconds; tests replace it.
        """
        self._enabled = enabled
        self._limiter = RateLimiter(
            {
                "ask": (
                    Rule(asks_per_minute, 60, "You're asking questions too quickly. Please try again in {wait}."),
                    Rule(
                        asks_per_day, 86_400, "You've reached today's limit of questions. Please try again in {wait}."
                    ),
                ),
                "load": (
                    Rule(
                        loads_per_hour,
                        3_600,
                        "You've loaded a lot of pages in a short time. Please try again in {wait}.",
                    ),
                ),
            },
            clock,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> ClientLimits:
        """Create the limits from the settings."""
        return cls(
            asks_per_minute=settings.rate_limit_asks_per_minute,
            asks_per_day=settings.rate_limit_asks_per_day,
            loads_per_hour=settings.rate_limit_loads_per_hour,
            enabled=settings.rate_limit_enabled,
        )

    def enforce(self, request: Request, action: str) -> None:
        """Count one use of ``action`` by the caller, or refuse it.

        Args:
            request: The incoming request.
            action: ``"ask"`` or ``"load"``.

        Raises:
            TooManyRequestsError: When the caller is over a limit. Its message says how long to wait.
        """
        if not self._enabled:
            return
        refusal = self._limiter.check(action, client_id(request))
        if refusal is not None:
            raise TooManyRequestsError(refusal.rule.message.format(wait=describe_wait(refusal.wait)), refusal.wait)
