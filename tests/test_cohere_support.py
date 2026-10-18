"""Tests of what the Cohere clients have in common, without any network."""

from typing import Any

import cohere
import pytest
from pydantic import SecretStr

from chat_with_any_website_agent.cohere_support import close_client
from chat_with_any_website_agent.embeddings import CohereEmbedder
from chat_with_any_website_agent.reranking import CohereReranker


class ContextClient:
    """A client that can be used in a ``with`` statement and records that it was left."""

    def __init__(self) -> None:
        self.left: list[tuple[Any, ...]] = []

    def __enter__(self) -> ContextClient:
        return self

    def __exit__(self, *details: Any) -> None:
        self.left.append(details)


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
