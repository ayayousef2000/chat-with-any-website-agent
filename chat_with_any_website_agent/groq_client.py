"""Chat completions with Groq, over several keys used in turn.

Several keys can be given, one for each account. The first is used until Groq says its limit is reached, then the
second, and so on; after the last, the first is used again. A key whose limit was reached rests for as long as Groq
asked and is skipped until it has recovered. The key in use stays in use until its own limit is reached.

Every key also has its own pace: a request goes to a ready key that has room for it in the minute, and waits only
when every key is full. Every attempt is counted, and so are the tokens of each reply, so that a run can be stopped at
its budget. Only the number of a key is ever logged, never the key.
"""

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import groq
from groq import Groq

from chat_with_any_website_agent.errors import BudgetExceededError
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy, retry_after_seconds
from chat_with_any_website_agent.throttle import Throttle
from chat_with_any_website_agent.usage import GROQ

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.usage import UsageCounter

logger = logging.getLogger(__name__)

# A limit, an outage or a connection that failed or timed out. The others (a wrong key, a bad request) would fail the
# same way again.
GROQ_RETRYABLE: tuple[type[BaseException], ...] = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
)

# How long a key is left alone after its limit was reached, when Groq does not say when it will work again.
DEFAULT_REST_SECONDS = 60.0

# The size of a request is estimated before it is sent, to pace the keys: a token is about three characters of text,
# and the reply is as long as the request allows or, when it is not limited, this many tokens. The real use that Groq
# reports afterwards corrects the count.
CHARS_PER_TOKEN = 3
DEFAULT_OUTPUT_TOKENS = 1024


# The error code of a request in JSON mode whose reply was not valid JSON.
JSON_FAILED_CODE = "json_validate_failed"


def _failed_json(error: groq.BadRequestError) -> str | None:
    """What the model produced when Groq refused it as invalid JSON, or ``None`` for any other bad request."""
    body = error.body
    detail = body.get("error") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("code") == JSON_FAILED_CODE:
        return str(detail.get("failed_generation") or "").strip()
    return None


def patient_retrier() -> Retrier:
    """A retrier for a run that may wait for a limit to lift, such as an evaluation, instead of a person who waits."""
    return Retrier(GROQ_RETRYABLE, RetryPolicy(attempts=5, max_delay=60.0, budget=240.0))


@dataclass(frozen=True)
class ChatReply:
    """The reply of the model to a request.

    Attributes:
        text: The text of the reply.
        prompt_tokens: The tokens the request used.
        completion_tokens: The tokens the reply used.
        key_number: Which key answered, counting from 1. The key itself is never exposed.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    key_number: int

    @property
    def total_tokens(self) -> int:
        """The tokens the request and the reply used together."""
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(messages: Sequence[Any], max_output_tokens: int) -> int:
    """Estimate how many tokens a request will use, before it is sent.

    Args:
        messages: The chat messages, each with a ``content``.
        max_output_tokens: The most tokens the reply may have.

    Returns:
        About a third of the characters of the messages, plus the reply allowance.
    """
    characters = sum(len(str(message.get("content") or "")) for message in messages)
    return characters // CHARS_PER_TOKEN + max_output_tokens


class GroqClient:
    """Sends chat requests to Groq with key rotation, a pace for each key, retries and usage counting."""

    def __init__(
        self,
        clients: Sequence[Any],
        model: str,
        tokens_per_minute: int,
        *,
        retrier: Retrier | None = None,
        usage: UsageCounter | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client over one Groq client for each key.

        Args:
            clients: One Groq client per key, in the order they are used. Their own retries should be off, because
                retrying is done here, once, with a policy that suits a person who is waiting.
            model: The model to ask, for example ``openai/gpt-oss-120b``.
            tokens_per_minute: The limit of each key (each account has its own).
            retrier: How to try again after a limit or an outage.
            usage: Counts every call and the tokens of every reply, and stops a run at its limits, when given.
            clock: The time, replaceable for tests.
            sleep: How to pause, replaceable for tests.

        Raises:
            ValueError: If there is no client.
        """
        if not clients:
            raise ValueError("at least one Groq client is needed")
        self._clients = list(clients)
        self._model = model
        self._retrier = retrier or Retrier(retryable=GROQ_RETRYABLE)
        self._usage = usage
        self._clock = clock
        self._throttles = [Throttle(tokens_per_minute, 60.0, clock=clock, sleep=sleep) for _ in self._clients]
        self._lock = threading.Lock()
        self._current = 0  # the key in use
        self._rested_until = [0.0] * len(self._clients)  # per key: the time from which it may be used again

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: str | None = None,
        usage: UsageCounter | None = None,
        retrier: Retrier | None = None,
    ) -> Self:
        """Create a client for the keys, the model and the pace in the settings.

        Args:
            settings: The settings. The main key comes first, then the backup keys, without blanks or repeats.
            model: The model to ask instead of the one in the settings, for example a judge model. Groq counts the
                limits for each model, so a client for another model has a pace of its own.
            usage: Counts every call and the tokens of every reply, when given.
            retrier: How to try again after a limit or an outage.

        Returns:
            The client.
        """
        clients = [Groq(api_key=key.get_secret_value(), max_retries=0) for key in settings.groq_keys]
        return cls(clients, model or settings.groq_model, settings.groq_tokens_per_minute, usage=usage, retrier=retrier)

    @property
    def model(self) -> str:
        """The model this client asks."""
        return self._model

    def close(self) -> None:
        """Close the connections of the Groq clients, for example when a short command is done."""
        for client in self._clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def complete(self, messages: Sequence[Any], **options: Any) -> ChatReply:
        """Send messages to the model and return its reply.

        Args:
            messages: The conversation to send, as chat messages with a ``role`` and a ``content``.
            **options: Further request options, such as ``temperature``, ``response_format`` or
                ``max_completion_tokens``.

        Returns:
            The reply, with the tokens it used.

        Raises:
            ServiceBusyError: If every key stays limited, or Groq stays unreachable, after the retries.
            BudgetExceededError: If the run's limit of calls or tokens is reached.
        """
        max_output = int(options.get("max_completion_tokens") or options.get("max_tokens") or DEFAULT_OUTPUT_TOKENS)
        estimate = estimate_tokens(messages, max_output)
        return self._retrier.call(lambda: self._request(messages, options, estimate), name="Groq")

    def _request(self, messages: Sequence[Any], options: dict[str, Any], estimate: int) -> ChatReply:
        """Make one request, moving on to the next key each time a key's limit is reached.

        Raises:
            groq.RateLimitError: When no key can be used. It is the error of the key that recovers soonest, so the
                wait reported to the person is the shortest one.
        """
        now = self._clock()
        candidates = self._candidates(now)
        soonest: tuple[float, groq.RateLimitError] | None = None
        paced_out: list[int] = []
        for position, index in enumerate(candidates):
            if not self._throttles[index].try_acquire(estimate):
                paced_out.append(index)  # a key that is fine but has no room in this minute
                continue
            try:
                return self._send(index, messages, options, estimate)
            except groq.RateLimitError as error:
                soonest = self._rest(index, error, soonest, next_index=candidates[position + 1 :])
        for index in paced_out:  # every ready key is limited or full: wait for the first that has a place free
            self._throttles[index].acquire(estimate)
            try:
                return self._send(index, messages, options, estimate)
            except groq.RateLimitError as error:
                soonest = self._rest(index, error, soonest, next_index=[])
        if soonest is None:  # pragma: no cover - there is always a candidate, and only a limit error leaves the loops
            raise AssertionError("unreachable")
        raise soonest[1]

    def _candidates(self, now: float) -> list[int]:
        """The keys to try, in order: the one in use, then the next ones, leaving out those that are resting."""
        with self._lock:
            count = len(self._clients)
            circle = [(self._current + step) % count for step in range(count)]
            ready = [index for index in circle if self._rested_until[index] <= now]
            # With every key resting, try the one that recovers first: its limit may have been lifted already.
            return ready or [min(circle, key=self._rested_until.__getitem__)]

    def _rest(
        self,
        index: int,
        error: groq.RateLimitError,
        soonest: tuple[float, groq.RateLimitError] | None,
        *,
        next_index: list[int],
    ) -> tuple[float, groq.RateLimitError]:
        """Let a key rest after its limit was reached, and remember the error of the key that recovers soonest."""
        asked = retry_after_seconds(error)
        rest = DEFAULT_REST_SECONDS if asked is None else asked
        with self._lock:
            self._rested_until[index] = self._clock() + rest  # from now, which may be later than the request began
        if next_index:
            logger.warning(
                "Groq key %d of %d reached its limit; trying key %d", index + 1, len(self._clients), next_index[0] + 1
            )
        return (rest, error) if soonest is None or rest < soonest[0] else soonest

    def _send(self, index: int, messages: Sequence[Any], options: dict[str, Any], estimate: int) -> ChatReply:
        if self._usage is not None:
            self._usage.ensure_tokens_left(GROQ)
            self._usage.record_call(GROQ)
        try:
            completion = self._clients[index].chat.completions.create(
                model=self._model, messages=list(messages), **options
            )
        except groq.BadRequestError as error:
            failed = _failed_json(error)
            if failed is None:
                raise
            # In JSON mode Groq refuses a reply that is not valid JSON, and an empty one is not. The reply is
            # handed on as the text it was, which every caller already treats as unreadable.
            logger.warning("Groq key %d could not make valid JSON for a request", index + 1)
            with self._lock:
                self._current = index
            return ChatReply(text=failed, prompt_tokens=0, completion_tokens=0, key_number=index + 1)
        with self._lock:
            self._current = index
        used = completion.usage
        reply = ChatReply(
            text=(completion.choices[0].message.content or "").strip(),
            prompt_tokens=int(used.prompt_tokens) if used else 0,
            completion_tokens=int(used.completion_tokens) if used else 0,
            key_number=index + 1,
        )
        if reply.total_tokens > estimate:
            self._throttles[index].record(reply.total_tokens - estimate)
        if self._usage is not None:
            # The reply that crossed the limit is still delivered; the next request is refused before it is sent.
            with contextlib.suppress(BudgetExceededError):
                self._usage.record_tokens(GROQ, reply.total_tokens)
        return reply
