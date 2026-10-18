"""Preparing a source: download it, read its text, split it, embed it and store it.

This runs before the agent answers, not as a step of its graph: a source is prepared once and then questions are
asked about the stored chunks. A source that is already stored is reused, so asking a second question costs no
embedding. A stored source is first checked, though: the website is asked whether the page changed since the copy was
made, and the copy is made again only if it did. So an answer is based on the page as it is when the question is
asked, and an unchanged page costs one small request.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, Self
from urllib.parse import urlsplit

from chat_with_any_website_agent.chunking import chunk_text, contextualize
from chat_with_any_website_agent.cleaning import clean_text
from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError
from chat_with_any_website_agent.extraction import NO_TEXT_MESSAGE, ExtractedPage, extract_page
from chat_with_any_website_agent.fetching import (
    FetchedDocument,
    Validators,
    fetch_document,
    fetch_if_changed,
    normalize_url,
)
from chat_with_any_website_agent.pdf import extract_pdf
from chat_with_any_website_agent.vector_store import SourceRecord, StoredChunk

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chat_with_any_website_agent.config import Settings

logger = logging.getLogger(__name__)

# The size of a chunk in characters and the overlap between neighbours, as in the original pipeline.
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300

# What can go wrong with a source without anything being wrong with the services. When the page of a stored source
# cannot be checked for one of these reasons, the stored copy is used.
SOURCE_ERRORS = (InvalidURLError, FetchError, ExtractionError)


class SourceStore(Protocol):
    """What ingestion needs from the store; ``WeaviateStore`` provides it."""

    @property
    def keeps_loaded_at(self) -> bool:
        """Whether the store keeps when each source was loaded."""
        ...

    @property
    def keeps_versions(self) -> bool:
        """Whether the store keeps which version of a page each copy is."""
        ...

    def has_source(self, url: str) -> bool:
        """Check whether any chunks are stored for a source."""
        ...

    def source_record(self, url: str) -> SourceRecord | None:
        """What is kept about the stored copy of a source, or ``None`` when it is not stored."""
        ...

    def replace_source(
        self, url: str, title: str, chunks: Sequence[StoredChunk], record: SourceRecord | None = None
    ) -> None:
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
        reused: ``True`` when the stored copy was used and nothing was embedded.
        title: The title of the source, or ``None`` when it was reused.
        chunk_count: How many chunks were stored, or ``None`` when it was reused.
        age: How long ago the stored copy that was reused was made, or ``None`` when that is not known or the source
            was loaded.
        expired: ``True`` when a stored copy was loaded again because it was older than the limit, or because its
            age or version was not known.
        checked: ``True`` when the website was asked and the page had not changed, so the stored copy was reused.
        changed: ``True`` when the page had changed since the stored copy was made, so it was loaded again.
        problem: Why the page of a stored copy could not be checked, in which case the stored copy was used.
    """

    url: str
    reused: bool
    title: str | None = None
    chunk_count: int | None = None
    age: timedelta | None = None
    expired: bool = False
    checked: bool = False
    changed: bool = False
    problem: str | None = None


@dataclass(frozen=True)
class _Prepared:
    """A downloaded document that was read: the page, its cleaned text and what the website said about its version."""

    page: ExtractedPage
    text: str
    validators: Validators

    @property
    def fingerprint(self) -> str:
        """A fingerprint of the cleaned text.

        The cleaned text is used and not the page, so that what changes on every visit, such as tracking codes, does
        not make an unchanged page look changed.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


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
        revalidate: Callable[[str, Validators], FetchedDocument | None] | None = None,
        check_interval: timedelta = timedelta(0),
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
            max_age: How old a stored copy may be before it is loaded again, or ``None`` for no limit. It has no
                effect when the store does not keep when a source was loaded.
            revalidate: Downloads a source again only if it changed, for example ``fetch_if_changed`` with its
                limits. It returns ``None`` when the website says that the stored version is current. Without it,
                a stored copy is not checked.
            check_interval: How long after a page was checked it is not checked again by this ingestor. Zero checks
                a page at every question.
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
        self._revalidate_fetch = revalidate
        self._check_interval = check_interval
        self._now = now
        self._checked: dict[str, datetime] = {}  # when each page was last checked, by this ingestor

    @classmethod
    def from_settings(cls, settings: Settings, store: SourceStore, embedder: DocumentEmbedder) -> Self:
        """Create an ingestor with the fetching, PDF, age and check limits of the settings.

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

        def revalidate(url: str, marks: Validators) -> FetchedDocument | None:
            return fetch_if_changed(
                url,
                marks,
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
            revalidate=revalidate,
            check_interval=timedelta(seconds=settings.source_check_seconds),
        )

    def ingest(self, url: str, *, refresh: bool = False, check: bool = True) -> IngestResult:
        """Make a source ready for questions, with the page as it is now.

        A stored copy is checked first: the website is asked whether the page changed since the copy was made. If it
        did not, the copy is reused. If it did, the copy is made again. If the website cannot be asked, the stored copy
        is used and the result says why. A copy that is older than the age limit, or whose age or version is not
        known because it was stored before the store kept them, is loaded again. Nothing is stored unless every step
        before it worked, so a failed attempt leaves an earlier copy as it was.

        Args:
            url: The address of a web page or PDF file.
            refresh: Download the source again even when it is already stored and has not changed.
            check: Ask the website whether a stored page changed. When ``False`` a stored copy is used as it is.

        Returns:
            What happened: the stored copy was reused (with its age when it is known, and whether the page was
            checked), or the source was stored with its title and number of chunks.

        Raises:
            InvalidURLError: If the address is invalid or not public.
            FetchError: If the source cannot be downloaded.
            ExtractionError: If it has no readable text.
            ServiceBusyError: If Cohere or Weaviate keeps failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        url = normalize_url(url)
        expired = changed = False
        prepared: _Prepared | None = None
        if not refresh and self._store.has_source(url):
            record = self._store.source_record(url) or SourceRecord()
            age = self._age(record)
            if self._is_too_old(age):
                expired = True
                logger.info("A stored source on %s is older than the limit and is loaded again", urlsplit(url).hostname)
            elif (revalidate := self._revalidate_fetch) is None or not self._should_check(url, check):
                return IngestResult(url=url, reused=True, age=age)
            else:
                outcome = self._revalidate(revalidate, url, record, age)
                if isinstance(outcome, IngestResult):
                    return outcome
                prepared = outcome
                changed = record.content_hash is not None
                expired = not changed  # a copy with no fingerprint is loaded again once, to learn its version
        if prepared is None:
            prepared = self._prepare(self._fetch(url))
        return self._store_prepared(url, prepared, expired=expired, changed=changed)

    def _revalidate(
        self,
        revalidate: Callable[[str, Validators], FetchedDocument | None],
        url: str,
        record: SourceRecord,
        age: timedelta | None,
    ) -> IngestResult | _Prepared:
        """Ask the website whether the page changed; give the stored copy back when it did not or cannot be asked.

        Returns:
            The result to hand back when the stored copy is used, or the new version of the page when it changed (or
            when the stored copy has no fingerprint to compare with).
        """
        marks = Validators(etag=record.etag, last_modified=record.last_modified)
        try:
            document = revalidate(url, marks)
            self._checked[url] = self._now()
            if document is None:
                return IngestResult(url=url, reused=True, age=age, checked=True)
            prepared = self._prepare(document)
        except SOURCE_ERRORS as error:
            logger.warning("The page of a stored source on %s could not be checked: %s", urlsplit(url).hostname, error)
            return IngestResult(url=url, reused=True, age=age, problem=str(error))
        if record.content_hash is not None and prepared.fingerprint == record.content_hash:
            return IngestResult(url=url, reused=True, age=age, checked=True)  # sent again, but the text is the same
        return prepared

    def _should_check(self, url: str, check: bool) -> bool:
        """Whether the page of a stored source is to be asked about now."""
        if not check or not self._store.keeps_versions:
            return False
        last = self._checked.get(url)
        return last is None or self._now() - last >= self._check_interval

    def _prepare(self, document: FetchedDocument) -> _Prepared:
        """Read a downloaded document and clean its text."""
        page = self._read(document)
        text = clean_text(page.text)
        if not text.strip():
            raise ExtractionError(NO_TEXT_MESSAGE)
        return _Prepared(page=page, text=text, validators=document.validators)

    def _store_prepared(self, url: str, prepared: _Prepared, *, expired: bool, changed: bool) -> IngestResult:
        chunks = chunk_text(prepared.text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise ExtractionError(NO_TEXT_MESSAGE)
        title = prepared.page.title
        vectors = self._embedder.embed_documents([contextualize(title, chunk.heading, chunk.text) for chunk in chunks])
        record = SourceRecord(
            loaded_at=self._now(),
            etag=prepared.validators.etag,
            last_modified=prepared.validators.last_modified,
            content_hash=prepared.fingerprint,
        )
        self._store.replace_source(
            url,
            title,
            [
                StoredChunk(text=chunk.text, heading=chunk.heading, vector=vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            record,
        )
        self._checked[url] = self._now()  # what was just loaded is as fresh as a check
        logger.info("Stored %d chunks of a source on %s", len(chunks), urlsplit(url).hostname)
        return IngestResult(
            url=url, reused=False, title=title, chunk_count=len(chunks), expired=expired, changed=changed
        )

    def _age(self, record: SourceRecord) -> timedelta | None:
        return None if record.loaded_at is None else self._now() - record.loaded_at

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
