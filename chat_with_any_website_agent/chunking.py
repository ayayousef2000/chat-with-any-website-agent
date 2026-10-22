"""Splitting cleaned text into overlapping chunks that respect paragraph and sentence boundaries."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_HEADING = re.compile(r"(#{1,6}) +(.+)")
_PARAGRAPH_BREAK = re.compile(r"\n{2,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_JOINER = "\n"
# The start of a section of the text of a PDF file: the heading of a page, or of the note at the end.
_SECTION_START = re.compile(r"^(?=## (?:Page \d+|Note)$)", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """A piece of page text together with the heading of the section it belongs to.

    Attributes:
        text: The text of the chunk.
        heading: The heading of the section the chunk starts in, or an empty string before the first heading.
    """

    text: str
    heading: str


@dataclass(frozen=True)
class _Unit:
    text: str
    heading: str
    is_heading: bool


def _split_words(text: str, size: int) -> Iterator[str]:
    words: list[str] = []
    length = 0
    for word in text.split():
        if words and length + 1 + len(word) > size:
            yield " ".join(words)
            words, length = [], 0
        length += len(word) + (1 if words else 0)
        words.append(word)
    if words:
        yield " ".join(words)


def _units(text: str, size: int) -> Iterator[_Unit]:
    """Yield paragraphs (tagged with their section heading), splitting oversized ones by sentence, then word."""
    heading = ""
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        match = _HEADING.fullmatch(paragraph.split("\n", 1)[0]) if paragraph.startswith("#") else None
        if match:
            heading = match.group(2).strip()
        if len(paragraph) <= size:
            yield _Unit(paragraph, heading, is_heading=bool(match) and "\n" not in paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            if len(sentence) <= size:
                yield _Unit(sentence, heading, is_heading=False)
            else:
                for piece in _split_words(sentence, size):
                    yield _Unit(piece, heading, is_heading=False)


def _joined_length(units: list[_Unit]) -> int:
    return sum(len(unit.text) for unit in units) + len(_JOINER) * max(len(units) - 1, 0)


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Return the trailing units of a chunk that fit within the overlap budget."""
    tail: list[_Unit] = []
    for unit in reversed(units):
        if _joined_length([unit, *tail]) > overlap:
            break
        tail.insert(0, unit)
    return tail


def _to_chunk(units: list[_Unit]) -> Chunk:
    return Chunk(text=_JOINER.join(unit.text for unit in units), heading=units[0].heading)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split cleaned Markdown text into overlapping chunks.

    Chunks respect paragraph boundaries where possible, fall back to sentences and then words for oversized
    paragraphs, and never leave a heading as the last line of a chunk.

    Args:
        text: Cleaned page text.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of trailing characters repeated at the start of the next chunk.

    Returns:
        The chunks in reading order.
    """
    chunks: list[Chunk] = []
    current: list[_Unit] = []

    for unit in _units(text, chunk_size):
        only_headings = all(held.is_heading for held in current)
        # A lone heading is kept with the next unit, even if that slightly exceeds the size limit.
        if current and not only_headings and _joined_length([*current, unit]) > chunk_size:
            # Don't leave a heading dangling at the end of a chunk; move it to the next one.
            moved = [current.pop()] if len(current) > 1 and current[-1].is_heading else []
            chunks.append(_to_chunk(current))
            current = [*_overlap_tail(current, chunk_overlap), *moved]
            while current and _joined_length([*current, unit]) > chunk_size:
                current.pop(0)
        current.append(unit)

    if current:
        chunks.append(_to_chunk(current))
    return chunks


def chunk_pages(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split the text of a PDF file into chunks that never cross a page.

    The text of a PDF file is a ``## Page N`` section for each page, and possibly a ``## Note`` section at the end. Each
    of those sections is chunked on its own, so a chunk holds the text of one page, its heading is that page, and the
    overlap between chunks never repeats the end of the page before. Chunked as one text, a short page would be merged
    into the chunk of the page before it and be labelled with that page, and the overlap at the start of a chunk would
    repeat the end of the previous page under the heading of the previous page.

    Args:
        text: The text of a PDF file, in sections.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of trailing characters repeated at the start of the next chunk of the same page.

    Returns:
        The chunks in reading order.
    """
    chunks: list[Chunk] = []
    for section in _SECTION_START.split(text):
        if section.strip():
            chunks.extend(chunk_text(section, chunk_size, chunk_overlap))
    return chunks


def contextualize(title: str, heading: str, text: str) -> str:
    """Prefix a chunk with its page title and section so the embedding captures where it came from.

    Args:
        title: The title of the page.
        heading: The heading of the section the chunk belongs to, or an empty string.
        text: The text of the chunk.

    Returns:
        The chunk text, preceded by the title (and the heading when it differs from the title).
    """
    context = f"{title} > {heading}" if heading and heading != title else title
    return f"{context}\n\n{text}"
