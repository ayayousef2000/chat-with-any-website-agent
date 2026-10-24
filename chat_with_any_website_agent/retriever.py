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
# When a question gives several sources, each of them keeps at least this many of the excerpts (see ``guaranteed``),
# and the search for each source asks for at least this many candidates.
DEFAULT_FLOOR = 2
MIN_GUARANTEED = 2
MIN_GROUP_CANDIDATES = 5


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

    One search over several sources lets the source with the most matching chunks crowd out the others: a long PDF file
    can fill all the excerpts and leave none from a short page that the question also asks about. So when two or more
    sources are ``guaranteed``, each is searched on its own, the candidates are reranked together, and each of those
    sources keeps its best ``floor`` excerpts (fewer when there are not so many places) before the rest are filled with
    the best of all.
    """

    urls: list[str] | None = None
    guaranteed: list[str] = Field(default_factory=list)
    floor: int = Field(default=DEFAULT_FLOOR, ge=1)
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
        guaranteed: list[str] | None = None,
        floor: int = DEFAULT_FLOOR,
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
            guaranteed: The sources that must each keep ``floor`` of the excerpts, for example the pages that a question
                gives. It takes effect only when two or more of them are among ``urls``.
            floor: How many excerpts each guaranteed source keeps at least.
            candidates: How many chunks the search returns for the reranker to choose from.
            top_n: How many excerpts to return.
            alpha: The blend of the two searches; 1 is the vector search only and 0 the keyword search only.
            **kwargs: Further LangChain retriever settings, such as ``tags``.

        Raises:
            pydantic.ValidationError: If a number is out of range.
        """
        settings: dict[str, Any] = {
            "urls": urls,
            "guaranteed": guaranteed or [],
            "floor": floor,
            "candidates": candidates,
            "top_n": top_n,
            "alpha": alpha,
        }
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
        vector = self._embedder.embed_query(query)
        groups = self._groups()
        if groups:
            each = max(MIN_GROUP_CANDIDATES, self.candidates // len(groups))
            chunks = [chunk for group in groups for chunk in self._store.search(query, vector, each, self.alpha, group)]
        else:
            chunks = self._store.search(query, vector, self.candidates, self.alpha, self.urls)
        if not chunks:
            return []
        texts = [contextualize(chunk.title, chunk.heading, chunk.text) for chunk in chunks]
        if not groups:
            ranked = self._reranker.rerank(query, texts, self.top_n)
            return [document_from_chunk(chunks[item.index], item.score) for item in ranked]
        everything = self._reranker.rerank(query, texts, len(texts))
        kept = _keep_floor(everything, [chunk.url for chunk in chunks], self._guaranteed_urls(), self.floor, self.top_n)
        return [document_from_chunk(chunks[item.index], item.score) for item in kept]

    def _guaranteed_urls(self) -> list[str]:
        """The guaranteed sources that are searched, each once; fewer than two means that nothing is guaranteed."""
        wanted = dict.fromkeys(self.guaranteed)
        urls = [url for url in wanted if self.urls is not None and url in self.urls]
        return urls if len(urls) >= MIN_GUARANTEED else []

    def _groups(self) -> list[list[str]]:
        """What to search on its own: each guaranteed source, then all the other sources together.

        Empty when nothing is guaranteed, and then one search covers all the sources.
        """
        guaranteed = self._guaranteed_urls()
        if not guaranteed or self.urls is None:
            return []
        groups = [[url] for url in guaranteed]
        others = [url for url in self.urls if url not in guaranteed]
        return [*groups, others] if others else groups


def _keep_floor(
    ranked: Sequence[RankedDocument], urls: Sequence[str], guaranteed: Sequence[str], floor: int, top_n: int
) -> list[RankedDocument]:
    """Choose ``top_n`` of the ranked chunks so that each guaranteed source keeps its best ``floor``.

    Args:
        ranked: Every candidate, best first.
        urls: The source of each candidate, by the index in the candidates.
        guaranteed: The sources that keep a floor.
        floor: How many each of them keeps at least. It is lowered to what fits: ``top_n`` shared by the sources.
        top_n: How many to keep.

    Returns:
        The kept chunks, best first.
    """
    room = min(floor, top_n // len(guaranteed))
    count = dict.fromkeys(guaranteed, 0)
    chosen: set[int] = set()
    for item in ranked:
        url = urls[item.index]
        if url in count and count[url] < room:
            count[url] += 1
            chosen.add(item.index)
    for item in ranked:
        if len(chosen) >= top_n:
            break
        chosen.add(item.index)
    return [item for item in ranked if item.index in chosen]


def document_from_chunk(chunk: RetrievedChunk, rerank_score: float | None) -> Document:
    """A chunk of a stored source as a LangChain document, with its place and scores in the metadata."""
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
