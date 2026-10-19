"""Tests of what the Cohere clients have in common, without any network."""

from typing import Any

import cohere
import pytest
from pydantic import SecretStr

from chat_with_any_website_agent.cohere_support import (
    COHERE_RETRYABLE,
    close_client,
    default_retrier,
    patient_cohere_retrier,
)
from chat_with_any_website_agent.embeddings import CohereEmbedder
from chat_with_any_website_agent.reranking import CohereReranker
from chat_with_any_website_agent.retrying import RetryPolicy


class ContextClient:
    """A client that can be used in a ``with`` statement and records that it was left."""

    def __init__(self) -> None:
        self.left: list[tuple[Any, ...]] = []

    def __enter__(self) -> ContextClient:
        return self

    def __exit__(self, *details: Any) -> None:
        self.left.append(details)


def test_a_patient_retrier_waits_for_a_limit_to_lift_longer_than_the_default() -> None:
    retrier = patient_cohere_retrier()

    assert retrier.retryable == COHERE_RETRYABLE
    assert retrier.policy == RetryPolicy(attempts=6, base_delay=10.0, max_delay=60.0, budget=300.0)
    assert retrier.policy.budget > default_retrier().policy.budget


def test_the_patient_retrier_pauses_for_a_minute_at_most_and_gives_up_within_its_budget() -> None:
    retrier = patient_cohere_retrier()
    pauses = [min(retrier.policy.base_delay * 2 ** (attempt - 1), retrier.policy.max_delay) for attempt in range(1, 6)]

    assert pauses == [10.0, 20.0, 40.0, 60.0, 60.0]
    assert sum(pauses) * 1.25 <= retrier.policy.budget  # even with the whole random spread added


def test_closing_a_client_leaves_its_with_statement() -> None:
    client = ContextClient()

    close_client(client)

    assert client.left == [(None, None, None)]


def test_a_client_that_cannot_be_used_in_a_with_statement_is_left_alone() -> None:
    close_client(object())  # nothing to close, and no error


def test_the_embedder_and_the_reranker_close_their_client() -> None:
    embedder_client, reranker_client = ContextClient(), ContextClient()
    embedder = CohereEmbedder(SecretStr("k"), "model-x", 8, client=embedder_client)
    reranker = CohereReranker(SecretStr("k"), "model-x", client=reranker_client)

    embedder.close()
    reranker.close()

    assert len(embedder_client.left) == 1
    assert len(reranker_client.left) == 1


@pytest.mark.filterwarnings("ignore:'asyncio.iscoroutinefunction' is deprecated:DeprecationWarning")
def test_closing_the_real_cohere_client_closes_its_connection() -> None:
    # Creating the client prints a notice from the Cohere library about a function of Python that is going away.
    client: Any = cohere.ClientV2(api_key="not-a-real-key")
    connection = client._client_wrapper.httpx_client.httpx_client
    assert not connection.is_closed

    close_client(client)

    assert connection.is_closed
