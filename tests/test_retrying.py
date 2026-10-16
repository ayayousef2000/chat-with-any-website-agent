"""Tests of waiting out rate limits and brief outages."""

import logging
from typing import TYPE_CHECKING

import httpx
import pytest

from chat_with_any_website_agent.errors import ServiceBusyError
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy, _spread, describe_wait, retry_after_seconds

if TYPE_CHECKING:
    from collections.abc import Callable


class TransientError(Exception):
    """A failure that is worth trying again, with the headers a service may send."""

    def __init__(self, **headers: str) -> None:
        super().__init__("transient")
        self.headers = headers


class LimitedError(Exception):
    """A failure that carries a whole response, as the errors of some clients do."""

    def __init__(self, **headers: str) -> None:
        super().__init__("limited")
        self.response = httpx.Response(429, headers=headers)


class FatalError(Exception):
    """A failure that trying again cannot fix."""


class Flaky:
    """An operation that raises the given errors one after the other and then succeeds."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _retrier(
    policy: RetryPolicy | None = None, jitter: Callable[[float], float] = lambda _: 0.0
) -> tuple[Retrier, list[float]]:
    pauses: list[float] = []
    retrier = Retrier(
        retryable=(TransientError, LimitedError), policy=policy or RetryPolicy(), sleep=pauses.append, jitter=jitter
    )
    return retrier, pauses


# --- the retry rules ---


def test_a_call_that_works_is_made_once_without_waiting() -> None:
    operation = Flaky()
    retrier, pauses = _retrier()

    assert retrier.call(operation) == "ok"
    assert (operation.calls, pauses) == (1, [])


def test_without_a_hint_the_pause_doubles_each_time() -> None:
    operation = Flaky(TransientError(), TransientError())
    retrier, pauses = _retrier()

    assert retrier.call(operation) == "ok"
    assert operation.calls == 3
    assert pauses == [1.0, 2.0]


def test_the_pause_the_service_asks_for_is_used() -> None:
    retrier, pauses = _retrier()

    assert retrier.call(Flaky(TransientError(**{"retry-after": "3"}))) == "ok"
    assert pauses == [3.0]


def test_a_pause_in_milliseconds_is_understood() -> None:
    retrier, pauses = _retrier()

    retrier.call(Flaky(TransientError(**{"retry-after-ms": "1500"})))

    assert pauses == [1.5]


def test_the_doubling_stops_at_the_longest_allowed_pause() -> None:
    retrier, pauses = _retrier(RetryPolicy(attempts=6, base_delay=4, max_delay=10, budget=100))

    retrier.call(Flaky(*[TransientError() for _ in range(5)]))

    assert pauses == [4.0, 8.0, 10.0, 10.0, 10.0]


def test_a_little_randomness_is_added_to_each_pause() -> None:
    retrier, pauses = _retrier(jitter=lambda delay: delay * 0.1)

    retrier.call(Flaky(TransientError()))

    assert pauses == [pytest.approx(1.1)]
    assert all(0 <= _spread(4) <= 1 for _ in range(50))


def test_after_the_last_attempt_the_service_is_reported_busy_with_the_wait_it_named() -> None:
    last = TransientError(**{"retry-after": "1"})
    operation = Flaky(TransientError(), TransientError(), TransientError(), last)
    retrier, pauses = _retrier(RetryPolicy(attempts=4))

    with pytest.raises(ServiceBusyError) as raised:
        retrier.call(operation, name="Cohere reranking")

    assert raised.value.__cause__ is last
    assert raised.value.retry_after == 1.0
    assert str(raised.value) == "Cohere reranking is busy right now. Try again in about 1 second."
    assert operation.calls == 4
    assert len(pauses) == 3


def test_a_service_that_asks_for_a_longer_wait_than_allowed_is_not_waited_for() -> None:
    operation = Flaky(TransientError(**{"retry-after": "50"}))
    retrier, pauses = _retrier(RetryPolicy(max_delay=20))

    with pytest.raises(ServiceBusyError) as raised:
        retrier.call(operation)

    assert raised.value.retry_after == 50.0
    assert (operation.calls, pauses) == (1, [])


def test_it_stops_when_the_total_waiting_time_would_pass_the_budget() -> None:
    operation = Flaky(TransientError(), TransientError(), TransientError())
    retrier, pauses = _retrier(RetryPolicy(attempts=5, base_delay=2, max_delay=20, budget=3))

    with pytest.raises(ServiceBusyError) as raised:
        retrier.call(operation)

    assert pauses == [2.0]  # the next pause of 4 seconds would pass the budget of 3
    assert raised.value.retry_after == 4.0


def test_an_error_that_trying_again_cannot_fix_is_raised_at_once_and_unchanged() -> None:
    operation = Flaky(FatalError("a wrong key"))
    retrier, pauses = _retrier()

    with pytest.raises(FatalError):
        retrier.call(operation)

    assert (operation.calls, pauses) == (1, [])


def test_each_retry_is_logged_with_what_was_called(caplog: pytest.LogCaptureFixture) -> None:
    retrier, _ = _retrier()

    with caplog.at_level(logging.WARNING, logger="chat_with_any_website_agent.retrying"):
        retrier.call(Flaky(TransientError()), name="Cohere embeddings")

    assert "Cohere embeddings failed with TransientError" in caplog.text
    assert "attempt 2 of 4" in caplog.text


# --- reading what the service asked for ---


def test_the_wait_is_read_from_the_error_and_from_its_response() -> None:
    assert retry_after_seconds(TransientError(**{"retry-after": "23"})) == 23.0
    assert retry_after_seconds(LimitedError(**{"retry-after": "7"})) == 7.0


def test_milliseconds_win_over_seconds_and_names_may_be_any_case() -> None:
    assert retry_after_seconds(TransientError(**{"Retry-After": "9", "RETRY-AFTER-MS": "2500"})) == 2.5


@pytest.mark.parametrize(
    "error",
    [TransientError(), TransientError(**{"retry-after": "soon"}), ValueError("no headers at all"), LimitedError()],
)
def test_no_wait_is_reported_when_none_was_named(error: BaseException) -> None:
    assert retry_after_seconds(error) is None


def test_a_negative_wait_becomes_zero() -> None:
    assert retry_after_seconds(TransientError(**{"retry-after": "-5"})) == 0.0


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, "a few seconds"),
        (0.2, "about 1 second"),
        (1, "about 1 second"),
        (23.4, "about 23 seconds"),
        (89, "about 89 seconds"),
        (200, "a few minutes"),
    ],
)
def test_waits_are_described_in_words(seconds: float | None, text: str) -> None:
    assert describe_wait(seconds) == text
