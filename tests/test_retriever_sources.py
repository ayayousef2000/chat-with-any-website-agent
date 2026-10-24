"""Tests of the retriever when a question gives several sources: each keeps a floor of the excerpts."""

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from chat_with_any_website_agent.reranking import RankedDocument
from chat_with_any_website_agent.retriever import (
    DEFAULT_CANDIDATES,
    DEFAULT_TOP_N,
    MIN_GROUP_CANDIDATES,
    SourceRetriever,
)
from chat_with_any_website_agent.vector_store import RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

PDF = "https://example.invalid/paper.pdf"
PAGE = "https://example.invalid/page"
OTHER = "https://example.invalid/other"
NAMES = {PDF: "pdf", PAGE: "page", OTHER: "other"}


def _chunk(url: str, index: int) -> RetrievedChunk:
    return RetrievedChunk(url=url, title="Doc", heading="", text=f"{NAMES[url]} {index}", chunk_index=index, score=0.4)


def _chunks(**counts: int) -> dict[str, list[RetrievedChunk]]:
    """The stored chunks: ``pdf=8`` stores 8 chunks of the PDF file."""
    return {url: [_chunk(url, index) for index in range(counts.get(name, 0))] for url, name in NAMES.items()}


class SourceStore:
    """Returns the chunks of the sources it is asked about, and records each search."""

    def __init__(self, chunks: dict[str, list[RetrievedChunk]]) -> None:
        self.chunks = chunks
        self.searches: list[dict[str, Any]] = []

    def search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float, urls: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append({"limit": limit, "urls": list(urls or [])})
        return [chunk for url in urls or [] for chunk in self.chunks.get(url, [])][:limit]


class FakeEmbedder:
    """Gives every question the same vector."""

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class Ranking:
    """Ranks the texts that start with ``first`` above the others, and otherwise keeps the order of the search."""

    def __init__(self, first: str = "") -> None:
        self.first = first
        self.requests: list[tuple[list[str], int]] = []

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        self.requests.append((list(documents), top_n))
        order = sorted(range(len(documents)), key=lambda index: (self.first not in documents[index], index))
        return [RankedDocument(index, 1.0 - rank / 100) for rank, index in enumerate(order[:top_n])]


def _retriever(
    chunks: dict[str, list[RetrievedChunk]], ranking: Ranking, **options: Any
) -> tuple[SourceRetriever, SourceStore]:
    store = SourceStore(chunks)
    return SourceRetriever(store, FakeEmbedder(), ranking, **options), store


def _urls(documents: Sequence[Any]) -> list[str]:
    return [document.metadata["url"] for document in documents]


def test_each_given_source_is_searched_on_its_own_and_the_candidates_are_reranked_together() -> None:
    ranking = Ranking()
    retriever, store = _retriever(_chunks(pdf=8, page=3), ranking, urls=[PDF, PAGE], guaranteed=[PDF, PAGE])

    retriever.invoke("Question?")

    assert store.searches == [{"limit": 10, "urls": [PDF]}, {"limit": 10, "urls": [PAGE]}]
    ((documents, top_n),) = ranking.requests
    assert len(documents) == 11  # all 8 chunks of the PDF file and all 3 of the page are reranked
    assert top_n == 11


def test_a_source_that_one_search_would_crowd_out_keeps_its_best_excerpts() -> None:
    retriever, _ = _retriever(_chunks(pdf=8, page=3), Ranking(first="pdf"), urls=[PDF, PAGE], guaranteed=[PDF, PAGE])

    documents = retriever.invoke("Question?")

    assert _urls(documents) == [PDF, PDF, PDF, PAGE, PAGE]  # best first
    assert [document.metadata["chunk_index"] for document in documents if document.metadata["url"] == PAGE] == [0, 1]


def test_the_same_ranking_leaves_the_page_out_when_nothing_is_guaranteed() -> None:
    retriever, store = _retriever(_chunks(pdf=8, page=3), Ranking(first="pdf"), urls=[PDF, PAGE])

    documents = retriever.invoke("Question?")

    assert set(_urls(documents)) == {PDF}
    assert store.searches == [{"limit": DEFAULT_CANDIDATES, "urls": [PDF, PAGE]}]


def test_one_guaranteed_source_changes_nothing() -> None:
    ranking = Ranking()
    retriever, store = _retriever(_chunks(pdf=8, page=3), ranking, urls=[PDF, PAGE], guaranteed=[PAGE])

    retriever.invoke("Question?")

    assert store.searches == [{"limit": DEFAULT_CANDIDATES, "urls": [PDF, PAGE]}]
    assert ranking.requests[0][1] == DEFAULT_TOP_N


def test_a_guaranteed_source_that_is_not_searched_is_ignored() -> None:
    retriever, store = _retriever(_chunks(pdf=8, page=3), Ranking(), urls=[PDF], guaranteed=[PDF, PAGE])

    retriever.invoke("Question?")

    assert store.searches == [{"limit": DEFAULT_CANDIDATES, "urls": [PDF]}]


def test_nothing_guaranteed_when_all_the_sources_are_searched() -> None:
    retriever, store = _retriever(_chunks(pdf=8, page=3), Ranking(), guaranteed=[PDF, PAGE])

    retriever.invoke("Question?")

    assert store.searches == [{"limit": DEFAULT_CANDIDATES, "urls": []}]  # urls=None is passed on as it is


def test_the_other_sources_are_searched_together_and_only_compete() -> None:
    retriever, store = _retriever(
        _chunks(pdf=8, page=3, other=3), Ranking(), urls=[PDF, PAGE, OTHER], guaranteed=[PDF, PAGE]
    )

    retriever.invoke("Question?")

    assert [search["urls"] for search in store.searches] == [[PDF], [PAGE], [OTHER]]


def test_a_source_that_has_fewer_excerpts_than_the_floor_keeps_what_it_has() -> None:
    retriever, _ = _retriever(
        _chunks(pdf=8, page=1), Ranking(first="pdf"), urls=[PDF, PAGE], guaranteed=[PDF, PAGE], floor=3
    )

    documents = retriever.invoke("Question?")

    assert len(documents) == DEFAULT_TOP_N
    assert _urls(documents).count(PAGE) == 1


def test_the_floor_is_lowered_to_what_the_excerpts_can_share() -> None:
    retriever, _ = _retriever(
        _chunks(pdf=8, page=3, other=3),
        Ranking(first="pdf"),
        urls=[PDF, PAGE, OTHER],
        guaranteed=[PDF, PAGE, OTHER],
        top_n=4,
        floor=2,
    )

    urls = _urls(retriever.invoke("Question?"))

    assert sorted(urls) == sorted([PDF, PDF, PAGE, OTHER])  # 4 places for 3 sources: the floor is 1 each


def test_more_guaranteed_sources_than_excerpts_leave_the_ranking_alone() -> None:
    retriever, _ = _retriever(
        _chunks(pdf=8, page=3, other=3),
        Ranking(first="pdf"),
        urls=[PDF, PAGE, OTHER],
        guaranteed=[PDF, PAGE, OTHER],
        top_n=2,
    )

    assert _urls(retriever.invoke("Question?")) == [PDF, PDF]


def test_a_search_for_many_sources_asks_for_a_few_candidates_from_each() -> None:
    chunks: dict[str, list[RetrievedChunk]] = {f"https://a.invalid/{number}": [] for number in range(4)}
    urls = list(chunks)
    store = SourceStore(chunks)
    retriever = SourceRetriever(store, FakeEmbedder(), Ranking(), urls=urls, guaranteed=urls, candidates=8)

    retriever.invoke("Question?")

    assert [search["limit"] for search in store.searches] == [MIN_GROUP_CANDIDATES] * 4


def test_nothing_is_reranked_when_none_of_the_given_sources_has_a_chunk() -> None:
    ranking = Ranking()
    retriever, _ = _retriever(_chunks(), ranking, urls=[PDF, PAGE], guaranteed=[PDF, PAGE])

    assert retriever.invoke("Question?") == []
    assert ranking.requests == []


def test_the_floor_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        _retriever(_chunks(), Ranking(), floor=0)
