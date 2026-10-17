"""Preparing a source: download it, read its text, split it, embed it and store it.

This runs before the agent answers, not as a step of its graph: a source is prepared once and then questions are
asked about the stored chunks. A source that is already stored is reused, so asking a second question costs no
download and no embedding.
"""

import logging
from dataclasses import dataclass
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

    def has_source(self, url: str) -> bool:
        """Check whether any chunks are stored for a source."""
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
    """

    url: str
    reused: bool
    title: str | None = None
    chunk_count: int | None = None


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
        )

    def ingest(self, url: str, *, refresh: bool = False) -> IngestResult:
        """Make a source ready for questions.

        Nothing is stored unless every step before it worked, so a failed attempt leaves an earlier copy as it was.

        Args:
            url: The address of a web page or PDF file.
            refresh: Download the source again even when it is already stored.

        Returns:
            What happened: the source was reused, or it was stored with its title and number of chunks.

        Raises:
            InvalidURLError: If the address is invalid or not public.
            FetchError: If the source cannot be downloaded.
            ExtractionError: If it has no readable text.
            ServiceBusyError: If Cohere or Weaviate keeps failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        url = normalize_url(url)
        if not refresh and self._store.has_source(url):
            return IngestResult(url=url, reused=True)

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
        return IngestResult(url=url, reused=False, title=page.title, chunk_count=len(chunks))

    def _read(self, document: FetchedDocument) -> ExtractedPage:
        if document.kind == "pdf":
            return extract_pdf(
                document.body, document.url, max_pages=self._pdf_max_pages, max_seconds=self._pdf_max_seconds
            )
        return extract_page(document.body, document.url)
