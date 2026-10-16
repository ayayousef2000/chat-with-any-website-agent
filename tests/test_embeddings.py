"""Tests of the Cohere embedder, with a fake client instead of the network."""

from types import SimpleNamespace
from typing import Any

import cohere.errors as cohere_errors
import httpx
import pytest
from pydantic import SecretStr

from chat_with_any_website_agent.cohere_support import COHERE_RETRYABLE
from chat_with_any_website_agent.embeddings import MAX_TEXTS_PER_CALL, CohereEmbedder
from chat_with_any_website_agent.errors import BudgetExceededError, ServiceBusyError
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy
from chat_with_any_website_agent.usage import COHERE, UsageCounter

DIMENSION = 8


class FakeCohere:
    """A stand-in for the Cohere client that records its calls and can fail the first ones."""

    def __init__(self, *failures: BaseException) -> None:
        self.failures = list(failures)
        self.calls: list[dict[str, Any]] = []

    def embed(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        vectors = [[0.5] * DIMENSION for _ in kwargs["texts"]]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))


def _embedder(
    client: Any, usage: UsageCounter | None = None, policy: RetryPolicy | None = None
) -> tuple[CohereEmbedder, list[float]]:
    pauses: list[float] = []
    retrier = Retrier(COHERE_RETRYABLE, policy or RetryPolicy(), sleep=pauses.append, jitter=lambda _: 0.0)
    embedder = CohereEmbedder(SecretStr("k"), "model-x", DIMENSION, retrier=retrier, usage=usage, client=client)
    return embedder, pauses


def test_documents_are_embedded_in_batches_with_the_document_input_type() -> None:
    client = FakeCohere()
    embedder, _ = _embedder(client)

    vectors = embedder.embed_documents([f"t{i}" for i in range(200)])

    assert len(vectors) == 200
    assert [len(call["texts"]) for call in client.calls] == [MAX_TEXTS_PER_CALL, MAX_TEXTS_PER_CALL, 8]
    assert {call["input_type"] for call in client.calls} == {"search_document"}
    assert client.calls[0]["model"] == "model-x"
    assert client.calls[0]["output_dimension"] == DIMENSION
    assert client.calls[0]["embedding_types"] == ["float"]


def test_a_question_is_embedded_with_the_query_input_type() -> None:
    client = FakeCohere()
    embedder, _ = _embedder(client)

    assert embedder.embed_query("hello") == [0.5] * DIMENSION
    assert client.calls[0]["input_type"] == "search_query"


def test_no_texts_means_no_call() -> None:
    client = FakeCohere()
    embedder, _ = _embedder(client)

    assert embedder.embed_documents([]) == []
    assert client.calls == []


def test_the_clients_own_retries_are_switched_off() -> None:
    client = FakeCohere()
    embedder, _ = _embedder(client)

    embedder.embed_query("hello")

    assert client.calls[0]["request_options"] == {"max_retries": 0}


@pytest.mark.parametrize(
    "answer",
    [
        SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.5] * DIMENSION])),  # one vector for two texts
        SimpleNamespace(embeddings=SimpleNamespace(float_=[])),
        SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.5] * 3, [0.5] * 3])),  # the wrong size
    ],
    ids=["wrong count", "no vectors", "wrong size"],
)
def test_an_answer_that_does_not_match_the_request_is_detected(answer: Any) -> None:
    client = SimpleNamespace(embed=lambda **kwargs: answer)
    embedder, _ = _embedder(client)

    with pytest.raises(RuntimeError, match="do not match the request"):
        embedder.embed_documents(["a", "b"])


# --- limits, outages and budgets ---


def test_a_rate_limit_is_waited_out_and_the_call_is_made_again() -> None:
    limit = cohere_errors.TooManyRequestsError(body={"message": "limit"}, headers={"retry-after": "4"})
    client = FakeCohere(limit)
    embedder, pauses = _embedder(client)

    assert embedder.embed_query("hello") == [0.5] * DIMENSION
    assert pauses == [4.0]
    assert len(client.calls) == 2


def test_each_batch_of_documents_is_retried_on_its_own() -> None:
    client = FakeCohere()
    embedder, pauses = _embedder(client)
    original = client.embed
    seen: list[int] = []

    def embed(**kwargs: Any) -> Any:
        seen.append(len(kwargs["texts"]))
        if len(seen) == 2:  # the second batch fails once
            raise cohere_errors.ServiceUnavailableError(body={"message": "down"})
        return original(**kwargs)

    client.embed = embed  # type: ignore[method-assign]

    assert len(embedder.embed_documents([f"t{i}" for i in range(150)])) == 150
    assert seen == [MAX_TEXTS_PER_CALL, 54, 54]  # the failed second batch was sent again, the first was not
    assert len(pauses) == 1


def test_a_connection_that_fails_is_tried_again() -> None:
    client = FakeCohere(httpx.ConnectError("no route"))
    embedder, pauses = _embedder(client)

    assert embedder.embed_query("hello") == [0.5] * DIMENSION
    assert len(pauses) == 1


def test_when_every_attempt_is_limited_the_caller_gets_a_busy_error_with_the_wait() -> None:
    limit = cohere_errors.TooManyRequestsError(body={"message": "limit"}, headers={"retry-after": "9"})
    client = FakeCohere(limit, limit, limit)
    embedder, pauses = _embedder(client, policy=RetryPolicy(attempts=3))

    with pytest.raises(ServiceBusyError) as raised:
        embedder.embed_query("hello")

    assert raised.value.retry_after == 9.0
    assert str(raised.value) == "Cohere embeddings is busy right now. Try again in about 9 seconds."
    assert len(pauses) == 2


def test_an_error_that_retrying_cannot_fix_reaches_the_caller_at_once() -> None:
    client = FakeCohere(cohere_errors.UnauthorizedError(body={"message": "invalid token"}))
    embedder, pauses = _embedder(client)

    with pytest.raises(cohere_errors.UnauthorizedError):
        embedder.embed_query("hello")

    assert (len(client.calls), pauses) == (1, [])


def test_every_attempt_is_counted_including_the_one_that_is_tried_again() -> None:
    usage = UsageCounter()
    client = FakeCohere(cohere_errors.ServiceUnavailableError(body={"message": "down"}))
    embedder, _ = _embedder(client, usage=usage)

    embedder.embed_query("hello")

    assert usage.calls == {COHERE: 2}


def test_a_run_stops_at_its_limit_of_calls() -> None:
    usage = UsageCounter(max_calls={COHERE: 2})
    client = FakeCohere()
    embedder, _ = _embedder(client, usage=usage)

    with pytest.raises(BudgetExceededError):
        embedder.embed_documents([f"t{i}" for i in range(200)])  # three batches, but only two calls are allowed

    assert len(client.calls) == 2
