"""Tests of the Cohere reranker, with a fake client instead of the network."""

from types import SimpleNamespace
from typing import Any

import cohere.errors as cohere_errors
import pytest
from pydantic import SecretStr

from chat_with_any_website_agent.cohere_support import COHERE_RETRYABLE
from chat_with_any_website_agent.errors import BudgetExceededError, ServiceBusyError
from chat_with_any_website_agent.reranking import CohereReranker, RankedDocument
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy
from chat_with_any_website_agent.throttle import Throttle
from chat_with_any_website_agent.usage import COHERE, UsageCounter

DOCUMENTS = ["text 0", "text 1", "text 2"]


class FakeCohere:
    """A stand-in for the Cohere client that records its calls and can fail the first ones."""

    def __init__(self, *failures: BaseException) -> None:
        self.failures = list(failures)
        self.calls: list[dict[str, Any]] = []

    def rerank(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        results = [SimpleNamespace(index=2, relevance_score=0.9), SimpleNamespace(index=0, relevance_score=0.5)]
        return SimpleNamespace(results=results)


class Clock:
    """A clock that only moves when something sleeps, so the tests never really wait."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _reranker(
    client: Any,
    throttle: Throttle | None = None,
    usage: UsageCounter | None = None,
    policy: RetryPolicy | None = None,
) -> tuple[CohereReranker, list[float]]:
    pauses: list[float] = []
    retrier = Retrier(COHERE_RETRYABLE, policy or RetryPolicy(), sleep=pauses.append, jitter=lambda _: 0.0)
    reranker = CohereReranker(
        SecretStr("k"), "rerank-model", throttle=throttle, retrier=retrier, usage=usage, client=client
    )
    return reranker, pauses


def test_texts_come_back_in_the_reranked_order_with_their_scores() -> None:
    client = FakeCohere()
    reranker, _ = _reranker(client)

    result = reranker.rerank("question", DOCUMENTS, top_n=2)

    assert result == [RankedDocument(index=2, score=0.9), RankedDocument(index=0, score=0.5)]
    call = client.calls[0]
    assert call["model"] == "rerank-model"
    assert call["query"] == "question"
    assert call["documents"] == DOCUMENTS
    assert call["top_n"] == 2
    assert call["request_options"] == {"max_retries": 0}


def test_top_n_is_not_larger_than_the_number_of_texts() -> None:
    client = FakeCohere()
    reranker, _ = _reranker(client)

    reranker.rerank("question", DOCUMENTS, top_n=50)

    assert client.calls[0]["top_n"] == 3


@pytest.mark.parametrize(
    ("documents", "expected"),
    [([], []), (["only one"], [RankedDocument(index=0, score=None)])],
    ids=["none", "one"],
)
def test_with_one_text_or_none_there_is_nothing_to_reorder_and_no_call(
    documents: list[str], expected: list[RankedDocument]
) -> None:
    client = FakeCohere()
    reranker, _ = _reranker(client)

    assert reranker.rerank("question", documents, top_n=5) == expected
    assert client.calls == []


# --- the pace, limits, outages and budgets ---


def test_requests_are_paced_to_the_limit_per_minute() -> None:
    clock = Clock()
    throttle = Throttle(2, 60, clock=clock, sleep=clock.sleep)
    reranker, _ = _reranker(FakeCohere(), throttle=throttle)

    for _ in range(3):
        reranker.rerank("question", DOCUMENTS, top_n=2)

    assert clock.now == 60.0  # the third request had to wait for the first to leave the window


def test_a_rate_limit_is_waited_out_and_the_call_is_made_again() -> None:
    limit = cohere_errors.TooManyRequestsError(body={"message": "limit"}, headers={"retry-after": "6"})
    client = FakeCohere(limit)
    reranker, pauses = _reranker(client)

    assert [item.index for item in reranker.rerank("question", DOCUMENTS, top_n=2)] == [2, 0]
    assert pauses == [6.0]
    assert len(client.calls) == 2


def test_when_every_attempt_is_limited_the_caller_gets_a_busy_error() -> None:
    limit = cohere_errors.TooManyRequestsError(body={"message": "limit"})
    reranker, _ = _reranker(FakeCohere(limit, limit), policy=RetryPolicy(attempts=2))

    with pytest.raises(ServiceBusyError) as raised:
        reranker.rerank("question", DOCUMENTS, top_n=2)

    assert str(raised.value) == "Cohere reranking is busy right now. Try again in a few seconds."


def test_a_retried_request_is_paced_again() -> None:
    clock = Clock()
    throttle = Throttle(1, 60, clock=clock, sleep=clock.sleep)
    limit = cohere_errors.TooManyRequestsError(body={"message": "limit"})
    reranker, _ = _reranker(FakeCohere(limit), throttle=throttle)

    reranker.rerank("question", DOCUMENTS, top_n=2)

    assert clock.now == 60.0  # the second attempt waited for the window of the first


def test_every_attempt_is_counted_and_a_run_stops_at_its_limit() -> None:
    usage = UsageCounter(max_calls={COHERE: 2})
    client = FakeCohere()
    reranker, _ = _reranker(client, usage=usage)

    reranker.rerank("question", DOCUMENTS, top_n=2)
    reranker.rerank("question", DOCUMENTS, top_n=2)
    with pytest.raises(BudgetExceededError):
        reranker.rerank("question", DOCUMENTS, top_n=2)

    assert len(client.calls) == 2
    assert usage.calls == {COHERE: 2}


def test_a_budget_that_is_used_up_does_not_make_the_caller_wait() -> None:
    clock = Clock()
    throttle = Throttle(1, 60, clock=clock, sleep=clock.sleep)
    usage = UsageCounter(max_calls={COHERE: 1})
    reranker, _ = _reranker(FakeCohere(), throttle=throttle, usage=usage)
    reranker.rerank("question", DOCUMENTS, top_n=2)

    with pytest.raises(BudgetExceededError):
        reranker.rerank("question", DOCUMENTS, top_n=2)

    assert clock.now == 0.0
