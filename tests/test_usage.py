"""Tests of the counting of calls and tokens, and of stopping a run at a limit."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from chat_with_any_website_agent.errors import BudgetExceededError
from chat_with_any_website_agent.usage import COHERE, GROQ, UsageCounter


def test_calls_are_counted_per_service() -> None:
    usage = UsageCounter()

    for _ in range(3):
        usage.record_call(COHERE)
    usage.record_call(GROQ)

    assert usage.calls == {COHERE: 3, GROQ: 1}


def test_without_a_limit_nothing_stops_a_run() -> None:
    usage = UsageCounter()

    for _ in range(1000):
        usage.record_call(COHERE)

    assert usage.calls[COHERE] == 1000
    assert usage.tokens_left(GROQ) is None


def test_a_run_stops_at_its_limit_of_calls_and_the_refused_call_is_not_counted() -> None:
    usage = UsageCounter(max_calls={COHERE: 2})
    usage.record_call(COHERE)
    usage.record_call(COHERE)

    with pytest.raises(BudgetExceededError) as raised:
        usage.record_call(COHERE)

    assert (raised.value.service, raised.value.limit) == (COHERE, 2)
    assert str(raised.value) == "The limit of 2 calls to cohere has been reached."
    assert usage.calls == {COHERE: 2}


def test_a_service_without_a_limit_is_not_stopped_by_another_service_limit() -> None:
    usage = UsageCounter(max_calls={COHERE: 1})
    usage.record_call(COHERE)

    usage.record_call(GROQ)
    usage.record_call(GROQ)

    assert usage.calls == {COHERE: 1, GROQ: 2}


def test_tokens_are_counted_and_the_limit_stops_the_run_with_true_totals() -> None:
    usage = UsageCounter(max_tokens={GROQ: 1000})
    usage.record_tokens(GROQ, 600)
    assert usage.tokens_left(GROQ) == 400

    with pytest.raises(BudgetExceededError) as raised:
        usage.record_tokens(GROQ, 500)

    assert (raised.value.service, raised.value.limit) == (GROQ, 1000)
    assert usage.tokens == {GROQ: 1100}  # counted first, so the total stays true
    assert usage.tokens_left(GROQ) == 0


def test_a_call_is_refused_before_it_is_sent_once_the_tokens_are_used_up() -> None:
    usage = UsageCounter(max_tokens={GROQ: 100})
    usage.ensure_tokens_left(GROQ)  # nothing used yet
    usage.record_tokens(GROQ, 99)
    usage.ensure_tokens_left(GROQ)  # one token is left

    usage.record_tokens(GROQ, 1)
    with pytest.raises(BudgetExceededError) as raised:
        usage.ensure_tokens_left(GROQ)

    assert (raised.value.service, raised.value.limit) == (GROQ, 100)


def test_a_service_without_a_token_limit_is_never_refused() -> None:
    usage = UsageCounter()
    usage.record_tokens(GROQ, 10_000_000)

    usage.ensure_tokens_left(GROQ)


def test_counting_is_exact_when_several_threads_call_at_once() -> None:
    usage = UsageCounter()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: usage.record_call(COHERE), range(800)))

    assert usage.calls[COHERE] == 800
