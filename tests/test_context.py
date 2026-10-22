"""Tests of finding the chunks next to the best excerpts, with a fake store instead of Weaviate."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from chat_with_any_website_agent.chunking import chunk_pages
from chat_with_any_website_agent.context import DEFAULT_BEST, DEFAULT_MAX_ADDED, ContextExpander
from chat_with_any_website_agent.pdf import extract_pdf
from chat_with_any_website_agent.vector_store import RetrievedChunk
from tests.pdf_files import make_pdf

if TYPE_CHECKING:
    from collections.abc import Sequence

A = "https://docs.example/a"
B = "https://docs.example/b"


def _chunk(url: str, index: int, heading: str = "Intro") -> RetrievedChunk:
    return RetrievedChunk(url=url, title="Title", heading=heading, text=f"text {index}", chunk_index=index, score=None)


def _document(url: str, index: int, heading: str = "Intro") -> Document:
    metadata = {"url": url, "title": "Title", "heading": heading, "chunk_index": index}
    return Document(page_content=f"text {index}", metadata=metadata)


@dataclass
class FakeStore:
    """Holds the chunks of the sources by place, and records what was asked."""

    chunks: dict[str, dict[int, RetrievedChunk]]
    asked: list[tuple[str, list[int]]] = field(default_factory=list)

    def fetch_chunks(self, url: str, indexes: Sequence[int]) -> list[RetrievedChunk]:
        self.asked.append((url, list(indexes)))
        held = self.chunks.get(url, {})
        return [held[index] for index in sorted(set(indexes)) if index in held]


def _source(url: str, count: int) -> dict[str, dict[int, RetrievedChunk]]:
    return {url: {index: _chunk(url, index) for index in range(count)}}


def _places(documents: Sequence[Document]) -> list[tuple[str, int]]:
    return [(str(d.metadata["url"]), int(d.metadata["chunk_index"])) for d in documents]


# --- the chunks that are added ---


def test_the_chunk_before_and_the_chunk_after_the_best_excerpt_are_added() -> None:
    store = FakeStore(_source(A, 10))

    added = ContextExpander(store).around([_document(A, 5)])

    assert _places(added) == [(A, 4), (A, 6)]  # the one before comes first
    assert [d.page_content for d in added] == ["text 4", "text 6"]


def test_an_added_chunk_has_the_place_title_and_heading_of_its_own_chunk() -> None:
    store = FakeStore({A: {4: _chunk(A, 4, "Earlier section"), 5: _chunk(A, 5), 6: _chunk(A, 6, "Later section")}})

    added = ContextExpander(store).around([_document(A, 5)])

    assert [d.metadata["heading"] for d in added] == ["Earlier section", "Later section"]
    assert all(d.metadata["url"] == A and d.metadata["title"] == "Title" for d in added)
    assert all(d.metadata["search_score"] is None and d.metadata["rerank_score"] is None for d in added)


def test_the_neighbours_of_the_first_two_excerpts_are_added_nearest_to_the_best_first() -> None:
    store = FakeStore(_source(A, 20))
    excerpts = [_document(A, 10), _document(A, 3), _document(A, 15)]  # the third is not one of the best

    added = ContextExpander(store).around(excerpts)

    assert _places(added) == [(A, 9), (A, 11), (A, 2), (A, 4)]
    assert DEFAULT_BEST == 2


def test_a_neighbour_that_is_already_an_excerpt_is_not_added_again() -> None:
    store = FakeStore(_source(A, 10))

    added = ContextExpander(store).around([_document(A, 5), _document(A, 6)])

    assert _places(added) == [(A, 4), (A, 7)]  # 6 is an excerpt and 5 is an excerpt


def test_a_neighbour_that_two_excerpts_share_is_added_once() -> None:
    store = FakeStore(_source(A, 10))

    added = ContextExpander(store).around([_document(A, 4), _document(A, 6)])

    assert _places(added) == [(A, 3), (A, 5), (A, 7)]


def test_there_is_nothing_before_the_first_chunk_and_nothing_after_the_last() -> None:
    store = FakeStore(_source(A, 3))

    assert _places(ContextExpander(store).around([_document(A, 0)])) == [(A, 1)]
    assert _places(ContextExpander(store).around([_document(A, 2)])) == [(A, 1)]
    assert store.asked[0] == (A, [1])  # the place before the first chunk is never asked for


def test_a_source_that_is_one_chunk_has_nothing_to_add() -> None:
    assert ContextExpander(FakeStore(_source(A, 1))).around([_document(A, 0)]) == []


def test_no_excerpts_have_nothing_to_widen_and_the_store_is_not_asked() -> None:
    store = FakeStore(_source(A, 5))

    assert ContextExpander(store).around([]) == []
    assert store.asked == []


def test_at_most_a_few_chunks_are_added_and_the_nearest_ones_are_kept() -> None:
    store = FakeStore(_source(A, 40))

    added = ContextExpander(store, max_added=3).around([_document(A, 10), _document(A, 20)], reach=2)

    assert _places(added) == [(A, 9), (A, 11), (A, 19)]  # the neighbours of the best excerpt come first
    assert DEFAULT_MAX_ADDED == 4


def test_a_wider_reach_adds_the_chunks_two_places_away_after_the_nearer_ones() -> None:
    store = FakeStore(_source(A, 20))

    added = ContextExpander(store, max_added=10).around([_document(A, 10)], reach=2)

    assert _places(added) == [(A, 9), (A, 11), (A, 8), (A, 12)]


def test_the_excerpts_of_two_sources_are_widened_in_their_own_source() -> None:
    store = FakeStore({**_source(A, 10), **_source(B, 10)})

    added = ContextExpander(store).around([_document(A, 5), _document(B, 2)])

    assert _places(added) == [(A, 4), (A, 6), (B, 1), (B, 3)]
    assert sorted(store.asked) == [(A, [4, 6]), (B, [1, 3])]  # one request for each source


def test_a_place_that_the_store_does_not_have_is_left_out() -> None:
    store = FakeStore({A: {4: _chunk(A, 4), 5: _chunk(A, 5)}})  # no chunk 6

    assert _places(ContextExpander(store).around([_document(A, 5)])) == [(A, 4)]


# --- the same, with the chunks of a real PDF file ---


def test_the_chunks_next_to_an_excerpt_of_a_real_pdf_file_keep_their_own_page() -> None:
    document = make_pdf([[f"Text of page {number}. It is about topic {number}."] for number in range(1, 6)])
    page = extract_pdf(document, A)
    chunks = {
        index: RetrievedChunk(A, "T", chunk.heading, chunk.text, index, None)
        for index, chunk in enumerate(chunk_pages(page.text, 200, 20))
    }
    store = FakeStore({A: chunks})
    best = _document(A, 2)  # a chunk of page 3
    best.metadata["heading"] = chunks[2].heading

    added = ContextExpander(store).around([best])

    assert [d.metadata["heading"] for d in added] == ["Page 2", "Page 4"]  # the text next to it, with its own page
