"""Tests of finding the pages that a question names and of reading them, with fakes instead of the services."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from chat_with_any_website_agent.chunking import chunk_pages
from chat_with_any_website_agent.pages import (
    MAX_PAGES_ASKED,
    NOTE_HEADING,
    PAGE_CHUNK_LIMIT,
    PageReader,
    describe_pages,
    page_heading,
    pages_in_question,
)
from chat_with_any_website_agent.pdf import extract_pdf
from chat_with_any_website_agent.reranking import RankedDocument
from chat_with_any_website_agent.vector_store import RetrievedChunk
from tests.pdf_files import make_pdf

if TYPE_CHECKING:
    from collections.abc import Sequence

PAPER = "https://arxiv.example/pdf/1706.03762"
OTHER = "https://arxiv.example/pdf/2005.14165"
WEB = "https://docs.example/tutorial"
EN_DASH = chr(0x2013)

# --- the pages that a question names ---


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does page 5 say about the heads?", (5,)),
        ("What does Page 12 say?", (12,)),
        ("What is on page5?", (5,)),
        ("What does page 5.", (5,)),
        ("According to page 7, which optimizer was used?", (7,)),
        ("What do pages 3-5 describe?", (3, 4, 5)),
        ("What do pages 3 to 5 describe?", (3, 4, 5)),
        ("What do pages 3 through 5 describe?", (3, 4, 5)),
        (f"What do pages 3{EN_DASH}5 describe?", (3, 4, 5)),
        ("What do pages 3 - 5 describe?", (3, 4, 5)),
        ("Compare pages 3 and 7.", (3, 7)),
        ("Compare pages 3, 5 and 7.", (3, 5, 7)),
        ("Compare pages 3 & 7.", (3, 7)),
        ("What do pp. 3-4 say?", (3, 4)),
        ("What does p. 7 say?", (7,)),
        ("What is on the first page?", (1,)),
        ("What is on the 1st page?", (1,)),
        ("What is on page 5 and on the first page?", (5, 1)),
        ("What does page 5 say, and page 5 again?", (5,)),
        ("In Python 3.11, what does page 4 say?", (4,)),
        ("ماذا تقول الصفحة 5؟", (5,)),
        ("ماذا تقول الصفحات 3-5؟", (3, 4, 5)),
        ("ماذا تقول الصفحة ٥؟", (5,)),  # an Arabic-Indic digit
    ],
)
def test_the_pages_a_question_names_are_found(question: str, expected: tuple[int, ...]) -> None:
    assert pages_in_question(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What does the pass statement do?",
        "What page is the table on?",
        "What is on the homepage 5 times a day?",
        "How many layers does it have in 2023?",
        "What does page 0 say?",
        "What does page zero say?",
        "",
    ],
)
def test_a_question_that_names_no_page_has_no_pages(question: str) -> None:
    assert pages_in_question(question) == ()


def test_a_range_is_cut_so_that_a_question_cannot_ask_for_a_whole_document() -> None:
    assert pages_in_question("What do pages 1-100 say?") == tuple(range(1, MAX_PAGES_ASKED + 1))


def test_at_most_a_few_pages_are_named() -> None:
    assert pages_in_question("pages 1, 2, 3, 4, 5, 6 and 7") == (1, 2, 3, 4, 5)


def test_a_range_that_runs_backwards_names_its_two_ends() -> None:
    assert pages_in_question("What do pages 7-3 say?") == (7, 3)


def test_pages_are_described_in_words() -> None:
    assert describe_pages([5]) == "page 5"
    assert describe_pages([3, 5]) == "pages 3 and 5"
    assert describe_pages([3, 5, 7]) == "pages 3, 5 and 7"


def test_the_heading_of_a_page_is_the_one_the_pdf_reader_writes() -> None:
    assert page_heading(12) == "Page 12"


# --- reading the pages ---


def _chunk(url: str, heading: str, index: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(url=url, title="The paper", heading=heading, text=text, chunk_index=index, score=None)


@dataclass
class FakeStore:
    """Holds the sections of the stored sources by their address and heading, and records what was asked."""

    sections: dict[tuple[str, str], list[RetrievedChunk]]
    asked: list[tuple[str, str, int]] = field(default_factory=list)

    def fetch_section(self, url: str, heading: str, limit: int) -> list[RetrievedChunk]:
        self.asked.append((url, heading, limit))
        return self.sections.get((url, heading), [])


class FakeReranker:
    """Keeps the documents it was told to keep, best first, and records its request."""

    def __init__(self, keep: Sequence[int]) -> None:
        self.keep = keep
        self.requests: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        self.requests.append((query, list(documents), top_n))
        return [RankedDocument(index=index, score=1.0 - rank / 10) for rank, index in enumerate(self.keep)]


def _paper() -> FakeStore:
    return FakeStore(
        {
            (PAPER, "Page 1"): [_chunk(PAPER, "Page 1", 0, "Title and abstract.")],
            (PAPER, "Page 3"): [_chunk(PAPER, "Page 3", 4, "The encoder has N = 6 identical layers.")],
            (PAPER, "Page 5"): [
                _chunk(PAPER, "Page 5", 8, "Multi-head attention uses h = 8 heads."),
                _chunk(PAPER, "Page 5", 9, "Each head uses dimension 64."),
            ],
        }
    )


def test_a_source_without_pages_is_not_a_pdf_and_nothing_else_is_read() -> None:
    store = FakeStore({})

    result = PageReader(store).read([WEB], [5], "What does page 5 say?")

    assert not result.is_pdf
    assert result.documents == ()
    assert store.asked == [(WEB, "Page 1", 1)]  # one look, to tell whether the source has pages


def test_the_named_pages_are_read_in_the_order_of_the_document() -> None:
    store = _paper()

    result = PageReader(store).read([PAPER], [5, 3], "What do the pages say?")

    assert result.is_pdf
    assert [(d.metadata["heading"], d.metadata["chunk_index"]) for d in result.documents] == [
        ("Page 5", 8),
        ("Page 5", 9),
        ("Page 3", 4),
    ]  # the pages in the order they were named, each page in the order of its chunks
    assert result.missing == ()
    assert result.note == ""
    assert (PAPER, "Page 5", PAGE_CHUNK_LIMIT) in store.asked


def test_an_excerpt_of_a_page_has_the_place_and_title_of_its_source() -> None:
    result = PageReader(_paper()).read([PAPER], [3], "q")

    (document,) = result.documents
    assert document.page_content == "The encoder has N = 6 identical layers."
    assert document.metadata == {
        "url": PAPER,
        "title": "The paper",
        "heading": "Page 3",
        "chunk_index": 4,
        "search_score": None,
        "rerank_score": None,
    }


def test_a_page_without_text_is_named_as_missing_and_the_note_of_the_document_is_read() -> None:
    store = _paper()
    note = "No text can be selected on page 4, which may be scans or images."
    store.sections[(PAPER, NOTE_HEADING)] = [_chunk(PAPER, NOTE_HEADING, 20, note)]

    result = PageReader(store).read([PAPER], [3, 4], "q")

    assert [d.metadata["heading"] for d in result.documents] == ["Page 3"]
    assert result.missing == (4,)
    assert result.note == note


def test_the_note_is_only_read_when_a_page_is_missing() -> None:
    store = _paper()

    PageReader(store).read([PAPER], [3], "q")

    assert all(heading != NOTE_HEADING for _, heading, _ in store.asked)


def test_a_page_that_one_of_several_documents_has_is_not_missing() -> None:
    store = _paper()
    store.sections[(OTHER, "Page 1")] = [_chunk(OTHER, "Page 1", 0, "Another paper.")]
    store.sections[(OTHER, "Page 3")] = [_chunk(OTHER, "Page 3", 3, "Few-shot learning.")]

    result = PageReader(store).read([PAPER, OTHER, WEB], [3], "q")

    assert [d.metadata["url"] for d in result.documents] == [PAPER, OTHER]
    assert result.missing == ()


def test_pages_that_none_of_the_documents_has_are_missing() -> None:
    result = PageReader(_paper()).read([PAPER], [40], "q")

    assert result.is_pdf
    assert result.documents == ()
    assert result.missing == (40,)


def test_the_most_relevant_excerpts_are_kept_in_the_order_of_the_document_when_there_are_too_many() -> None:
    store = FakeStore(
        {
            (PAPER, "Page 1"): [_chunk(PAPER, "Page 1", 0, "t")],
            (PAPER, "Page 2"): [_chunk(PAPER, "Page 2", index, f"chunk {index}") for index in range(5)],
        }
    )
    reranker = FakeReranker(keep=[4, 1, 2])  # best first

    result = PageReader(store, reranker, max_excerpts=3).read([PAPER], [2], "What is the best chunk?")

    assert [d.page_content for d in result.documents] == ["chunk 1", "chunk 2", "chunk 4"]  # the order of the page
    assert [d.metadata["rerank_score"] for d in result.documents] == pytest.approx([0.9, 0.8, 1.0])
    [(query, texts, top_n)] = reranker.requests
    assert (query, top_n) == ("What is the best chunk?", 3)
    assert texts[0].startswith("The paper")  # shown with title and heading
    assert texts[0].endswith("chunk 0")


def test_nothing_is_reranked_when_the_pages_fit() -> None:
    reranker = FakeReranker(keep=[0])

    PageReader(_paper(), reranker).read([PAPER], [3, 5], "q")  # three excerpts, and five may be kept

    assert reranker.requests == []


def test_the_first_excerpts_are_kept_when_there_is_no_reranker() -> None:
    store = FakeStore(
        {
            (PAPER, "Page 1"): [_chunk(PAPER, "Page 1", 0, "t")],
            (PAPER, "Page 2"): [_chunk(PAPER, "Page 2", index, f"chunk {index}") for index in range(5)],
        }
    )

    result = PageReader(store, max_excerpts=2).read([PAPER], [2], "q")

    assert [d.page_content for d in result.documents] == ["chunk 0", "chunk 1"]


# --- the pages that the PDF reader writes are the ones the page reader looks for ---


def _stored(pdf_bytes: bytes) -> FakeStore:
    """The chunks that ingesting a real PDF file would store, held by their heading as the store holds them."""
    page = extract_pdf(pdf_bytes, PAPER)
    sections: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for index, chunk in enumerate(chunk_pages(page.text, 200, 20)):
        sections.setdefault((PAPER, chunk.heading), []).append(_chunk(PAPER, chunk.heading, index, chunk.text))
    return FakeStore(sections)


def test_a_page_of_a_real_pdf_file_is_found_by_its_number_and_no_other_page_is() -> None:
    document = make_pdf([[f"Text of page {number}. It is about topic {number}."] for number in range(1, 13)])

    result = PageReader(_stored(document)).read([PAPER], [10], "What is on page 10?")

    assert result.is_pdf
    assert [d.metadata["heading"] for d in result.documents] == ["Page 10"]  # not "Page 1", "Page 11" or "Page 12"
    assert "topic 10" in result.documents[0].page_content
    assert result.missing == ()


def test_a_page_beyond_a_real_pdf_file_is_missing() -> None:
    document = make_pdf([[f"Text of page {number}."] for number in range(1, 4)])

    result = PageReader(_stored(document)).read([PAPER], [9], "What is on page 9?")

    assert result.is_pdf
    assert result.missing == (9,)
    assert result.documents == ()


def test_the_note_about_a_page_without_text_is_found_for_a_real_pdf_file() -> None:
    document = make_pdf([["Text one."], [], ["Text three."], ["Text four."]])

    result = PageReader(_stored(document)).read([PAPER], [2], "What is on page 2?")

    assert result.missing == (2,)
    assert result.note.startswith("No text can be selected on page 2")


def test_a_page_that_is_longer_than_a_chunk_is_read_whole_in_order() -> None:
    sentences = [f"Sentence number {number} of the long page says something useful." for number in range(12)]
    document = make_pdf([["Short first page."], sentences])

    result = PageReader(_stored(document), max_excerpts=20).read([PAPER], [2], "q")

    assert len(result.documents) > 1
    joined = " ".join(d.page_content for d in result.documents)
    assert joined.index("Sentence number 0 ") < joined.index("Sentence number 11 ")
    assert all(d.metadata["heading"] == "Page 2" for d in result.documents)
