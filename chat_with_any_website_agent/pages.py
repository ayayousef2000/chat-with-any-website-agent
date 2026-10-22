"""Questions about a page of a PDF file: finding the page numbers in a question and reading those pages.

A question such as "What does page 5 say about the heads?" names its page, but a search by meaning does not look at
page numbers, so it can answer from another page. Plain code finds the numbers, and the text of exactly those pages is
read from the store. Each page of a stored PDF file is a section with the heading ``Page N`` (see ``pdf.py``), so a
page is a section that can be read by its heading, in the order of the document. A source that has no such sections is
not a PDF file, and a number in a question about it is only a number.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from chat_with_any_website_agent.chunking import contextualize
from chat_with_any_website_agent.retriever import DEFAULT_TOP_N, document_from_chunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document

    from chat_with_any_website_agent.reranking import RankedDocument
    from chat_with_any_website_agent.vector_store import RetrievedChunk

# The most pages a question may name. A question about more pages than this is answered from the first ones.
MAX_PAGES_ASKED = 5

# The most chunks of one page that are read, which is far more than a page has.
PAGE_CHUNK_LIMIT = 20

# The heading of the section where the text of a PDF file notes its limits (pages that were not read, empty pages).
NOTE_HEADING = "Note"
NOTE_CHUNK_LIMIT = 3

# The words that name a page, in English and in Arabic ("page", "the page", "pages", "the pages"), and the words and
# signs that join page numbers into a range ("3-5", "3 to 5") or a list ("3 and 5", "3, 5").
_PAGE_WORDS = r"(?:pages?|pp?\.|الصفحة|صفحة|الصفحات|صفحات)"
_EN_DASH = chr(0x2013)
_EM_DASH = chr(0x2014)
_RANGE_SIGNS = ("-", _EN_DASH, _EM_DASH, "to", "through")
_JOINERS = rf"(?:-|{_EN_DASH}|{_EM_DASH}|to|through|and|&|,)"
_REFERENCE = re.compile(rf"(?<!\w){_PAGE_WORDS}\s*(\d+(?:\s*{_JOINERS}\s*\d+)*)", re.IGNORECASE)
_PART = re.compile(r"\d+|[^\d\s]+")
_FIRST_PAGE = re.compile(r"\b(?:first|1st) page\b", re.IGNORECASE)


def pages_in_question(question: str) -> tuple[int, ...]:
    """Find the pages that a question names: ``page 5``, ``pages 3-5``, ``pages 3 and 7``, ``p. 7``, ``the first page``.

    Args:
        question: The question.

    Returns:
        The page numbers in the order they are named, each once and from 1 upwards, and at most ``MAX_PAGES_ASKED`` of
        them. Empty when the question names no page.
    """
    pages: list[int] = []
    for match in _REFERENCE.finditer(question):
        previous: int | None = None
        joiner = ""
        for part in _PART.findall(match.group(1)):
            if not part[0].isdigit():
                joiner = part.casefold()
                continue
            number = _number(part)
            if joiner in _RANGE_SIGNS and previous is not None and previous < number:
                pages.extend(range(previous + 1, min(number, previous + MAX_PAGES_ASKED) + 1))
            else:
                pages.append(number)
            previous, joiner = number, ""
    if _FIRST_PAGE.search(question):
        pages.append(1)
    return tuple(dict.fromkeys(page for page in pages if page >= 1))[:MAX_PAGES_ASKED]


def _number(digits: str) -> int:
    """A number written with the digits of any script."""
    return int("".join(str(unicodedata.digit(character)) for character in digits))


class PageStore(Protocol):
    """What the page reader needs from the store; ``WeaviateStore`` provides it."""

    def fetch_section(self, url: str, heading: str, limit: int) -> list[RetrievedChunk]:
        """Read the chunks of one section of a stored source, in the order of the source."""
        ...


class PageReranker(Protocol):
    """What the page reader needs from the reranker; ``CohereReranker`` provides it."""

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        """Keep the texts most relevant to the question."""
        ...


@dataclass(frozen=True)
class PageRead:
    """What reading the named pages of the sources gave.

    Attributes:
        is_pdf: Whether any of the sources is a PDF file. When it is not, the numbers of the question are not pages
            and nothing was read.
        documents: The excerpts of the named pages, in the order of the document.
        missing: The named pages that none of the PDF files has any text for.
        note: What the stored text says about the limits of the document, for example that some pages have no text
            that can be selected or that only the first pages were read. Empty when it says nothing.
    """

    is_pdf: bool
    documents: tuple[Document, ...] = ()
    missing: tuple[int, ...] = ()
    note: str = ""


class PageReader:
    """Reads the pages of stored PDF files that a question names."""

    def __init__(
        self, store: PageStore, reranker: PageReranker | None = None, *, max_excerpts: int = DEFAULT_TOP_N
    ) -> None:
        """Create a reader.

        Args:
            store: Where the chunks are stored, for example a ``WeaviateStore``.
            reranker: Keeps the most relevant excerpts when the pages hold more than ``max_excerpts``. Without it the
                first ones are kept.
            max_excerpts: The most excerpts to return.
        """
        self._store = store
        self._reranker = reranker
        self._max_excerpts = max_excerpts

    def read(self, urls: Sequence[str], numbers: Sequence[int], query: str) -> PageRead:
        """Read the named pages of the PDF files among the sources.

        Args:
            urls: The sources of the question.
            numbers: The page numbers the question names.
            query: The question, to choose among the excerpts when the pages hold too many.

        Returns:
            The excerpts of the pages, the pages that have none, and the note of the document about its limits.

        Raises:
            ServiceBusyError: If Weaviate or Cohere keeps failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        files = [url for url in urls if self._store.fetch_section(url, page_heading(1), 1)]
        if not files:
            return PageRead(is_pdf=False)
        chunks: list[RetrievedChunk] = []
        found: set[int] = set()
        for url in files:
            for number in numbers:
                section = self._store.fetch_section(url, page_heading(number), PAGE_CHUNK_LIMIT)
                if section:
                    found.add(number)
                    chunks.extend(section)
        missing = tuple(number for number in numbers if number not in found)
        note = self._note(files) if missing else ""
        return PageRead(is_pdf=True, documents=tuple(self._best(chunks, query)), missing=missing, note=note)

    def _note(self, files: Sequence[str]) -> str:
        for url in files:
            chunks = self._store.fetch_section(url, NOTE_HEADING, NOTE_CHUNK_LIMIT)
            if chunks:
                lines = (line.strip() for chunk in chunks for line in chunk.text.split("\n"))
                return " ".join(line for line in lines if line and line != f"## {NOTE_HEADING}")
        return ""

    def _best(self, chunks: list[RetrievedChunk], query: str) -> list[Document]:
        """The excerpts to answer from: all of them, or the most relevant ones in the order of the document."""
        if len(chunks) <= self._max_excerpts or self._reranker is None:
            return [document_from_chunk(chunk, None) for chunk in chunks[: self._max_excerpts]]
        ranked = self._reranker.rerank(
            query, [contextualize(chunk.title, chunk.heading, chunk.text) for chunk in chunks], self._max_excerpts
        )
        return [document_from_chunk(chunks[item.index], item.score) for item in sorted(ranked, key=lambda r: r.index)]


def page_heading(number: int) -> str:
    """The heading that the text of a PDF file gives to a page."""
    return f"Page {number}"


def describe_pages(numbers: Sequence[int]) -> str:
    """Page numbers in words: ``page 5``, ``pages 3 and 5`` or ``pages 3, 5 and 7``."""
    if len(numbers) == 1:
        return f"page {numbers[0]}"
    shown = [str(number) for number in numbers]
    return f"pages {', '.join(shown[:-1])} and {shown[-1]}"
