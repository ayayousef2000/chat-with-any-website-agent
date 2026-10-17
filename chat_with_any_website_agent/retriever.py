"""A LangChain retriever over the stored chunks: a hybrid search, then reranking.

The retriever chains the three services of retrieval, so that a graph node, or any LangChain chain, can call
``retriever.invoke(question)`` and get the best excerpts as documents. The services are not imported here: anything
with the matching method works, which keeps the retriever easy to test.
"""

from typing import TYPE_CHECKING, Any, Protocol

# LangChain reads the signature of ``_get_relevant_documents`` at run time, so this import has to stay a real import.
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: TC002
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from chat_with_any_website_agent.chunking import contextualize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chat_with_any_website_agent.reranking import RankedDocument
    from chat_with_any_website_agent.vector_store import RetrievedChunk

# How many chunks the search returns for reranking, how many the reranker keeps, and the blend of the two searches
# (1 is the vector search only, 0 the keyword search only).
DEFAULT_CANDIDATES = 20
DEFAULT_TOP_N = 5
DEFAULT_ALPHA = 0.5


class ChunkSearch(Protocol):
    """What the retriever needs from the store; ``WeaviateStore`` provides it."""

    def search(
        self,
        query: str,
        vector: Sequence[float],
        limit: int,
        alpha: float,
        urls: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search the stored chunks."""
        ...


class QueryEmbedder(Protocol):
    """What the retriever needs from the embedder; ``CohereEmbedder`` provides it."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a question."""
        ...


class Reranker(Protocol):
    """What the retriever needs from the reranker; ``CohereReranker`` provides it."""

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        """Keep the texts most relevant to the question."""
        ...


class SourceRetriever(BaseRetriever):
    """Finds the excerpts of the stored sources that best answer a question.

    Every document holds the text of one chunk. Its metadata has the ``url`` and ``title`` of the source, the
    ``heading`` of its section, its ``chunk_index``, the ``search_score`` of the search and the ``rerank_score``
    (``None`` when there was nothing to reorder).
    """

    urls: list[str] | None = None
    candidates: int = Field(default=DEFAULT_CANDIDATES, ge=1)
    top_n: int = Field(default=DEFAULT_TOP_N, ge=1)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)

    _store: ChunkSearch = PrivateAttr()
    _embedder: QueryEmbedder = PrivateAttr()
    _reranker: Reranker = PrivateAttr()

    def __init__(
        self,
        store: ChunkSearch,
        embedder: QueryEmbedder,
        reranker: Reranker,
        *,
        urls: list[str] | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        top_n: int = DEFAULT_TOP_N,
        alpha: float = DEFAULT_ALPHA,
        **kwargs: Any,
    ) -> None:
        """Create a retriever over a store, an embedder and a reranker.

        Args:
            store: Where the chunks are stored, for example a ``WeaviateStore``.
            embedder: Turns the question into a vector, for example a ``CohereEmbedder``.
            reranker: Reorders the candidates, for example a ``CohereReranker``.
            urls: The sources to search in. ``None`` searches all of them, and an empty list finds nothing.
            candidates: How many chunks the search returns for the reranker to choose from.
            top_n: How many excerpts to return.
            alpha: The blend of the two searches; 1 is the vector search only and 0 the keyword search only.
            **kwargs: Further LangChain retriever settings, such as ``tags``.

        Raises:
            pydantic.ValidationError: If a number is out of range.
        """
        settings: dict[str, Any] = {"urls": urls, "candidates": candidates, "top_n": top_n, "alpha": alpha}
        super().__init__(**settings, **kwargs)  # as a dict, because mypy only knows the fields of the base class
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Search, rerank and return the best excerpts, best first.

        Raises:
            ServiceBusyError: If Cohere or Weaviate keeps failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        if self.urls is not None and not self.urls:
            return []
        chunks = self._store.search(query, self._embedder.embed_query(query), self.candidates, self.alpha, self.urls)
        if not chunks:
            return []
        ranked = self._reranker.rerank(
            query, [contextualize(chunk.title, chunk.heading, chunk.text) for chunk in chunks], self.top_n
        )
        return [_document(chunks[item.index], item.score) for item in ranked]


def _document(chunk: RetrievedChunk, rerank_score: float | None) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "url": chunk.url,
            "title": chunk.title,
            "heading": chunk.heading,
            "chunk_index": chunk.chunk_index,
            "search_score": chunk.score,
            "rerank_score": rerank_score,
        },
    )
