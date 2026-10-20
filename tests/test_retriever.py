"""Tests of the LangChain retriever, with fakes instead of Weaviate and Cohere."""

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from chat_with_any_website_agent.reranking import RankedDocument
from chat_with_any_website_agent.retriever import DEFAULT_ALPHA, DEFAULT_CANDIDATES, DEFAULT_TOP_N, SourceRetriever
from chat_with_any_website_agent.vector_store import RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

URL = "https://example.invalid/guide"


def _chunk(index: int, heading: str = "Setup", score: float | None = 0.4) -> RetrievedChunk:
    return RetrievedChunk(url=URL, title="Guide", heading=heading, text=f"text {index}", chunk_index=index, score=score)


class FakeStore:
    """Returns the chunks it was given and records how it was searched."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.searches: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        vector: Sequence[float],
        limit: int,
        alpha: float,
        urls: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.searches.append({"query": query, "vector": list(vector), "limit": limit, "alpha": alpha, "urls": urls})
        return self.chunks


class FakeEmbedder:
    """Gives every question the same vector and records the questions."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.questions.append(text)
        return [0.1, 0.2]


class FakeReranker:
    """Returns the ranking it was given and records what it was asked to reorder."""

    def __init__(self, ranking: list[RankedDocument]) -> None:
        self.ranking = ranking
        self.requests: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        self.requests.append((query, list(documents), top_n))
        return self.ranking


def _retriever(
    chunks: list[RetrievedChunk], ranking: list[RankedDocument], **options: Any
) -> tuple[SourceRetriever, FakeStore, FakeEmbedder, FakeReranker]:
    store, embedder, reranker = FakeStore(chunks), FakeEmbedder(), FakeReranker(ranking)
    return SourceRetriever(store, embedder, reranker, **options), store, embedder, reranker


def test_the_best_excerpts_come_back_as_documents_in_the_order_of_the_reranker() -> None:
    chunks = [_chunk(0), _chunk(1, heading="Usage", score=0.3), _chunk(2)]
    retriever, _, _, _ = _retriever(chunks, [RankedDocument(2, 0.9), RankedDocument(1, 0.5)])

    documents = retriever.invoke("How do I install it?")

    assert [document.page_content for document in documents] == ["text 2", "text 1"]
    assert documents[1].metadata == {
        "url": URL,
        "title": "Guide",
        "heading": "Usage",
        "chunk_index": 1,
        "search_score": 0.3,
        "rerank_score": 0.5,
    }


def test_the_question_is_embedded_and_searched_with_the_settings_of_the_retriever() -> None:
    retriever, store, embedder, _ = _retriever(
        [_chunk(0)], [RankedDocument(0, None)], urls=[URL], candidates=30, top_n=3, alpha=0.7
    )

    retriever.invoke("Question?")

    assert embedder.questions == ["Question?"]
    assert store.searches == [{"query": "Question?", "vector": [0.1, 0.2], "limit": 30, "alpha": 0.7, "urls": [URL]}]


def test_the_defaults_search_every_source_and_keep_five_excerpts() -> None:
    retriever, store, _, reranker = _retriever([_chunk(0)], [RankedDocument(0, None)])

    retriever.invoke("Question?")

    assert store.searches[0]["limit"] == DEFAULT_CANDIDATES
    assert store.searches[0]["alpha"] == DEFAULT_ALPHA
    assert store.searches[0]["urls"] is None
    assert reranker.requests[0][2] == DEFAULT_TOP_N == 5


def test_the_reranker_sees_each_chunk_with_the_title_and_section_it_came_from() -> None:
    retriever, _, _, reranker = _retriever([_chunk(0), _chunk(1, heading="")], [RankedDocument(0, 0.8)])

    retriever.invoke("Question?")

    question, texts, _ = reranker.requests[0]
    assert question == "Question?"
    assert texts == ["Guide > Setup\n\ntext 0", "Guide\n\ntext 1"]


def test_an_excerpt_without_a_rerank_score_keeps_the_score_of_the_search() -> None:
    retriever, _, _, _ = _retriever([_chunk(0, score=0.6)], [RankedDocument(0, None)])

    (document,) = retriever.invoke("Question?")

    assert document.metadata["search_score"] == 0.6
    assert document.metadata["rerank_score"] is None


def test_nothing_is_reranked_when_the_search_finds_nothing() -> None:
    retriever, _, _, reranker = _retriever([], [])

    assert retriever.invoke("Question?") == []
    assert reranker.requests == []


def test_an_empty_list_of_sources_finds_nothing_without_calling_a_service() -> None:
    retriever, store, embedder, reranker = _retriever([_chunk(0)], [RankedDocument(0, 0.9)], urls=[])

    assert retriever.invoke("Question?") == []
    assert embedder.questions == []
    assert store.searches == []
    assert reranker.requests == []


@pytest.mark.parametrize(
    "options",
    [{"candidates": 0}, {"top_n": 0}, {"alpha": -0.1}, {"alpha": 1.1}],
)
def test_a_number_out_of_range_is_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _retriever([], [], **options)
