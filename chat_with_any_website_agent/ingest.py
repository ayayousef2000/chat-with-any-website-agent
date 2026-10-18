"""Preparing a source: download it, read its text, split it, embed it and store it.

This runs before the agent answers, not as a step of its graph: a source is prepared once and then questions are
asked about the stored chunks. A source that is already stored is reused, so asking a second question costs no
download and no embedding.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, Self
from urllib.parse import urlsplit

from chat_with_any_website_agent.chunking import chunk_text, contextualize
from chat_with_any_website_agent.cleaning import clean_text
from chat_with_any_website_agent.errors import ExtractionError
from chat_with_any_website_agent.extraction import NO_TEXT_MESSAGE, ExtractedPage, extract_page
from chat_with_any_website_agent.fetching import FetchedDocument, fetch_document, normalize_url
from chat_with_any_website_agent.pdf import extract_pdf
from chat_with_any_website_agent.vector_store import StoredChunk

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chat_with_any_website_agent.config import Settings

logger = logging.getLogger(__name__)

# The size of a chunk in characters and the overlap between neighbours, as in the original pipeline.
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300


class SourceStore(Protocol):
    """What ingestion needs from the store; ``WeaviateStore`` provides it."""

    @property
    def keeps_loaded_at(self) -> bool:
        """Whether the store keeps when each source was loaded."""
        ...

    def has_source(self, url: str) -> bool:
        """Check whether any chunks are stored for a source."""
        ...

    def loaded_at(self, url: str) -> datetime | None:
        """When a source was loaded, or ``None`` when that is not known."""
        ...

    def replace_source(self, url: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        """Store the chunks of a source in place of any earlier ones."""
        ...


class DocumentEmbedder(Protocol):
    """What ingestion needs from the embedder; ``CohereEmbedder`` provides it."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts that will be stored and searched later."""
        ...


@dataclass(frozen=True)
class IngestResult:
    """What happened to a source.

    Attributes:
        url: The normalized address the source is stored under.
        reused: ``True`` when the source was already stored and nothing was downloaded.
        title: The title of the source, or ``None`` when it was reused.
        chunk_count: How many chunks were stored, or ``None`` when it was reused.
        age: How long ago the stored copy that was reused was made, or ``None`` when that is not known or the source
            was loaded.
        expired: ``True`` when a stored copy was loaded again because it was older than the limit, or because its
            age was not known.
    """

    url: str
    reused: bool
    title: str | None = None
    chunk_count: int | None = None
    age: timedelta | None = None
    expired: bool = False


class SourceIngestor:
    """Prepares web pages and PDF files for questions."""

    def __init__(
        self,
        store: SourceStore,
        embedder: DocumentEmbedder,
        fetch: Callable[[str], FetchedDocument],
        *,
        pdf_max_pages: int,
        pdf_max_seconds: float,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_age: timedelta | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Create an ingestor.

        Args:
            store: Where the chunks are stored, for example a ``WeaviateStore``.
            embedder: Turns the chunks into vectors, for example a ``CohereEmbedder``.
            fetch: Downloads a source, for example ``fetch_document`` with its limits.
            pdf_max_pages: The most pages of a PDF file to read.
            pdf_max_seconds: The longest time to spend reading a PDF file.
            chunk_size: The target size of a chunk in characters.
            chunk_overlap: How many characters of a chunk are repeated at the start of the next one.
            max_age: How old a stored copy may be before it is loaded again, or ``None`` to keep it until a refresh
                is asked for. It has no effect when the store does not keep when a source was loaded.
            now: The time, replaceable for tests.

        Raises:
            ValueError: If the sizes make no sense.
        """
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_size must be positive and chunk_overlap must be at least 0 and below it")
        self._store = store
        self._embedder = embedder
        self._fetch = fetch
        self._pdf_max_pages = pdf_max_pages
        self._pdf_max_seconds = pdf_max_seconds
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_age = max_age
        self._now = now

    @classmethod
    def from_settings(cls, settings: Settings, store: SourceStore, embedder: DocumentEmbedder) -> Self:
        """Create an ingestor with the fetching and PDF limits of the settings.

        Args:
            settings: The settings.
            store: Where the chunks are stored.
            embedder: Turns the chunks into vectors.

        Returns:
            The ingestor.
        """

        def fetch(url: str) -> FetchedDocument:
            return fetch_document(
                url,
                timeout=settings.fetch_timeout_seconds,
                max_bytes=settings.fetch_max_bytes,
                pdf_max_bytes=settings.pdf_max_bytes,
            )

        return cls(
            store,
            embedder,
            fetch,
            pdf_max_pages=settings.pdf_max_pages,
            pdf_max_seconds=settings.pdf_max_seconds,
            max_age=timedelta(hours=settings.source_max_age_hours) if settings.source_max_age_hours else None,
        )

    def ingest(self, url: str, *, refresh: bool = False) -> IngestResult:
        """Make a source ready for questions.

        A stored copy is reused while it is younger than the limit. A copy that is older, or whose age is not known
        because it was stored before the store kept ages, is loaded again. Nothing is stored unless every step before
        it worked, so a failed attempt leaves an earlier copy as it was.

        Args:
            url: The address of a web page or PDF file.
            refresh: Download the source again even when it is already stored and young.

        Returns:
            What happened: the source was reused, with its age when it is known, or it was stored with its title and
            number of chunks.

        Raises:
            InvalidURLError: If the address is invalid or not public.
            FetchError: If the source cannot be downloaded.
            ExtractionError: If it has no readable text.
            ServiceBusyError: If Cohere or Weaviate keeps failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        url = normalize_url(url)
        expired = False
        if not refresh and self._store.has_source(url):
            age = self._age(url)
            expired = self._is_too_old(age)
            if not expired:
                return IngestResult(url=url, reused=True, age=age)
            logger.info("A stored source on %s is older than the limit and is loaded again", urlsplit(url).hostname)

        page = self._read(self._fetch(url))
        chunks = chunk_text(clean_text(page.text), self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise ExtractionError(NO_TEXT_MESSAGE)
        vectors = self._embedder.embed_documents(
            [contextualize(page.title, chunk.heading, chunk.text) for chunk in chunks]
        )
        self._store.replace_source(
            url,
            page.title,
            [
                StoredChunk(text=chunk.text, heading=chunk.heading, vector=vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        logger.info("Stored %d chunks of a source on %s", len(chunks), urlsplit(url).hostname)
        return IngestResult(url=url, reused=False, title=page.title, chunk_count=len(chunks), expired=expired)

    def _age(self, url: str) -> timedelta | None:
        loaded = self._store.loaded_at(url) if self._store.keeps_loaded_at else None
        return None if loaded is None else self._now() - loaded

    def _is_too_old(self, age: timedelta | None) -> bool:
        """Whether a stored copy has to be loaded again: it is older than the limit, or its age is not known."""
        if self._max_age is None or not self._store.keeps_loaded_at:
            return False  # there is no limit, or the age of a copy cannot be told, so it is kept
        return age is None or age > self._max_age

    def _read(self, document: FetchedDocument) -> ExtractedPage:
        if document.kind == "pdf":
            return extract_pdf(
                document.body, document.url, max_pages=self._pdf_max_pages, max_seconds=self._pdf_max_seconds
            )
        return extract_page(document.body, document.url)
