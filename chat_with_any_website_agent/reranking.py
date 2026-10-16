"""Reordering texts by their relevance to a question with a Cohere reranker."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cohere

from chat_with_any_website_agent.cohere_support import NO_CLIENT_RETRIES, default_retrier
from chat_with_any_website_agent.usage import COHERE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import SecretStr

    from chat_with_any_website_agent.retrying import Retrier
    from chat_with_any_website_agent.throttle import Throttle
    from chat_with_any_website_agent.usage import UsageCounter


@dataclass(frozen=True)
class RankedDocument:
    """One text of a reranking, by its place in the list that was given.

    Attributes:
        index: The position of the text in the list given to ``rerank``.
        score: How relevant the text is to the question (0 to 1), or ``None`` when no reranking was needed.
    """

    index: int
    score: float | None


class CohereReranker:
    """Reorders texts by relevance to a question using the Cohere rerank API."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        throttle: Throttle | None = None,
        retrier: Retrier | None = None,
        usage: UsageCounter | None = None,
        client: Any = None,
    ) -> None:
        """Create a reranker.

        Args:
            api_key: The Cohere key.
            model: The rerank model, for example ``rerank-v4.0-fast``.
            throttle: Keeps the requests below the plan's limit per minute, when given.
            retrier: How to try again after a limit or an outage. The default suits a person who is waiting.
            usage: Counts every call and stops a run at its limit, when given.
            client: A client to use instead of the Cohere one, for tests.
        """
        self._client: Any = client if client is not None else cohere.ClientV2(api_key=api_key.get_secret_value())
        self._model = model
        self._throttle = throttle
        self._retrier = retrier or default_retrier()
        self._usage = usage

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        """Keep the texts most relevant to the question.

        With one text or none there is nothing to reorder, so Cohere is not called.

        Args:
            query: The question.
            documents: The candidate texts.
            top_n: How many to keep.

        Returns:
            Up to ``top_n`` texts, most relevant first.

        Raises:
            ServiceBusyError: If Cohere keeps refusing or failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        if len(documents) <= 1:
            return [RankedDocument(index=index, score=None) for index in range(len(documents))]

        def attempt() -> Any:
            if self._usage is not None:
                self._usage.record_call(COHERE)
            if self._throttle is not None:
                self._throttle.acquire()
            return self._client.rerank(
                model=self._model,
                query=query,
                documents=list(documents),
                top_n=min(top_n, len(documents)),
                request_options=NO_CLIENT_RETRIES,
            )

        response = self._retrier.call(attempt, name="Cohere reranking")
        return [RankedDocument(index=result.index, score=result.relevance_score) for result in response.results]
