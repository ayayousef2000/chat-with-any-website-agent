"""Reading the text of a PDF file, a page at a time.

Each page becomes a section with a ``## Page N`` heading, so an answer can cite the page it came from. Reading is
bounded in pages and in time, and a file that cannot be read gives a plain message and never a traceback.
"""

import logging
import time
from io import BytesIO
from itertools import islice
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

from pypdf import PdfReader

from chat_with_any_website_agent.errors import ExtractionError
from chat_with_any_website_agent.extraction import ExtractedPage

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

PDF_NO_TEXT_MESSAGE = (
    "This PDF has no text that can be selected. It may be a scan or a set of images, which cannot be read. "
    "Try a different document."
)
PDF_PROTECTED_MESSAGE = "This PDF is protected with a password, so it cannot be read. Try a different document."
PDF_UNREADABLE_MESSAGE = "This PDF could not be read. The file may be damaged. Try a different document."

DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_SECONDS = 30.0

# A PDF without a title of its own gets the biggest text of its first page, when it is this much bigger than the body
# text and reads like a title: words, and between these lengths.
TITLE_SIZE_FACTOR = 1.15
MIN_TITLE_CHARS = 8
MAX_TITLE_CHARS = 200


def extract_pdf(
    data: bytes, url: str, *, max_pages: int = DEFAULT_MAX_PAGES, max_seconds: float = DEFAULT_MAX_SECONDS
) -> ExtractedPage:
    """Read the text of a PDF file, a page at a time, each page under its own ``## Page N`` heading.

    The headings let an answer cite a page ("section: Page 12"). Only the first ``max_pages`` pages are read, and
    reading stops after ``max_seconds`` on a slow host. The text says how many pages were read when the document is
    longer, so an answer about a later page is not mistaken for a gap in the document.

    Args:
        data: The bytes of the PDF file.
        url: The address the file came from, used for the fallback title and for the log.
        max_pages: The most pages to read.
        max_seconds: The longest time to spend reading.

    Returns:
        The text of the document, its title, and its address.

    Raises:
        ExtractionError: If the file is not a readable PDF, is protected by a password or has no selectable text.
    """
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise ExtractionError(PDF_UNREADABLE_MESSAGE)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError(PDF_PROTECTED_MESSAGE)
        total = len(reader.pages)
        started = time.monotonic()
        read = 0
        sections = []
        first_page: list[tuple[float, str]] = []  # the text of page one with its font sizes, to find a title
        for number, page in enumerate(islice(reader.pages, max_pages), start=1):
            if time.monotonic() - started > max_seconds:
                break
            if number == 1:
                text = _reflow(page.extract_text(visitor_text=_collect_sizes(first_page)) or "")
            else:
                text = _reflow(page.extract_text() or "")
            read = number
            if text:
                sections.append(f"## Page {number}\n\n{text}")
        title = _pdf_title(reader, url, first_page)
    except ExtractionError:
        raise
    except Exception as exc:
        host = urlsplit(url).hostname or "an unknown host"  # the path of a file can hold private things
        logger.warning("Could not read a PDF from %s: %s", host, type(exc).__name__)
        raise ExtractionError(PDF_UNREADABLE_MESSAGE) from exc
    if not sections:
        raise ExtractionError(PDF_NO_TEXT_MESSAGE)
    if read < total:
        sections.append(f"## Note\n\nOnly the first {read} of the {total} pages of this document were read.")
    return ExtractedPage(url=url, title=title, text="\n\n".join(sections))


def _reflow(text: str) -> str:
    """Join the lines of a PDF page into paragraphs.

    A PDF stores text line by line as it is drawn, so a sentence is cut at the end of every line. Lines are joined
    with a space, a word that was split with a hyphen at the end of a line is joined without one, and a blank line
    stays a paragraph break.
    """
    paragraphs: list[str] = []
    current = ""
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
        elif current.endswith("-") and line[:1].islower() and current[-2:-1].isalpha():
            current = current[:-1] + line
        else:
            current = f"{current} {line}" if current else line
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _collect_sizes(into: list[tuple[float, str]]) -> Callable[..., None]:
    """A reader of text pieces for pypdf that records every piece of a page with its font size."""

    def visit(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        if text.strip():
            into.append((round(float(font_size or 0), 1), text.strip()))

    return visit


def _title_from_text(pieces: list[tuple[float, str]]) -> str:
    """The title of a paper or report: the biggest text on the first page that reads like words.

    The body is the font size with most text. A title is bigger than the body, so sizes are tried from the biggest
    down, and a size is skipped when its text is not made of words, such as a stamp of numbers and brackets
    (``arXiv:1706.03762v7 [cs.CL] 2 Aug 2023``), which an archive prints bigger than the title itself.
    """
    if not pieces:
        return ""
    characters: dict[float, int] = {}
    for size, text in pieces:
        characters[size] = characters.get(size, 0) + len(text)
    body = max(characters, key=lambda size: characters[size])
    for size in sorted((size for size in characters if size > body * TITLE_SIZE_FACTOR), reverse=True):
        candidate = " ".join(" ".join(text for piece_size, text in pieces if piece_size == size).split())
        letters = sum(character.isalpha() for character in candidate)
        words = len(candidate.split())
        if MIN_TITLE_CHARS <= len(candidate) <= MAX_TITLE_CHARS and words >= 2 and letters >= len(candidate) * 0.75:
            return candidate
    return ""


def _pdf_title(reader: PdfReader, url: str, first_page: list[tuple[float, str]]) -> str:
    """The title written into the PDF, else the biggest text of its first page, else the name of the file."""
    title = (reader.metadata.title if reader.metadata else None) or ""
    title = " ".join(str(title).split())
    if title:
        return title
    title = _title_from_text(first_page)
    if title:
        return title
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    return name or url
