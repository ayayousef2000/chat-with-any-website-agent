"""Turning text into vectors with Cohere."""

from typing import TYPE_CHECKING, Any, Literal

import cohere

from chat_with_any_website_agent.cohere_support import NO_CLIENT_RETRIES, close_client, default_retrier
from chat_with_any_website_agent.usage import COHERE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import SecretStr

    from chat_with_any_website_agent.retrying import Retrier
    from chat_with_any_website_agent.usage import UsageCounter

MAX_TEXTS_PER_CALL = 96

InputType = Literal["search_document", "search_query"]


class CohereEmbedder:
    """Creates text embeddings with the Cohere API."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        dimension: int,
        *,
        retrier: Retrier | None = None,
        usage: UsageCounter | None = None,
        client: Any = None,
    ) -> None:
        """Create an embedder.

        Args:
            api_key: The Cohere key.
            model: The embedding model, for example ``embed-v5.0-pro``.
            dimension: The size of the vectors. Every vector in one collection must have the same size.
            retrier: How to try again after a limit or an outage. The default suits a person who is waiting.
            usage: Counts every call and stops a run at its limit, when given.
            client: A client to use instead of the Cohere one, for tests.
        """
        self._client: Any = client if client is not None else cohere.ClientV2(api_key=api_key.get_secret_value())
        self._model = model
        self._dimension = dimension
        self._retrier = retrier or default_retrier()
        self._usage = usage

    def close(self) -> None:
        """Close the connection to Cohere, for example when a short command is done."""
        close_client(self._client)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts that will be stored and searched later.

        Args:
            texts: The texts to embed. They are sent in batches that respect the API limit.

        Returns:
            One vector per input text, in the same order.

        Raises:
            ServiceBusyError: If Cohere keeps refusing or failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_TEXTS_PER_CALL):
            vectors.extend(self._embed(texts[start : start + MAX_TEXTS_PER_CALL], "search_document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching stored documents.

        Args:
            text: The question.

        Returns:
            The query vector.

        Raises:
            ServiceBusyError: If Cohere keeps refusing or failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        return self._embed([text], "search_query")[0]

    def _embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        def attempt() -> Any:
            if self._usage is not None:
                self._usage.record_call(COHERE)
            return self._client.embed(
                model=self._model,
                texts=list(texts),
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self._dimension,
                request_options=NO_CLIENT_RETRIES,
            )

        response = self._retrier.call(attempt, name="Cohere embeddings")
        vectors = response.embeddings.float_
        if not vectors or len(vectors) != len(texts) or any(len(vector) != self._dimension for vector in vectors):
            raise RuntimeError("Cohere returned embeddings that do not match the request.")
        return list(vectors)
