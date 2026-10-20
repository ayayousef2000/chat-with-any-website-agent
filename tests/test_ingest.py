"""Tests of the ingestion of a source, with fakes instead of the network, Cohere and Weaviate."""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from chat_with_any_website_agent import ingest
from chat_with_any_website_agent.cleaning import clean_text
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError, ServiceBusyError
from chat_with_any_website_agent.extraction import extract_page
from chat_with_any_website_agent.fetching import FetchedDocument, Validators
from chat_with_any_website_agent.ingest import IngestResult, SourceIngestor
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.vector_store import SourceRecord
from tests.pdf_files import make_pdf

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chat_with_any_website_agent.vector_store import StoredChunk

URL = "https://example.com/a"

ARTICLE = b"""<html><head><title>Sample Article</title></head><body>
<nav><a href="/">Home</a></nav>
<article><h1>Sample Article</h1>
<p>Retrieval systems find relevant passages before a model writes an answer. This paragraph is long enough
for the extractor to treat it as real content rather than boilerplate on the page.</p>
<p>A second paragraph continues the article with more detail about how chunking and embeddings work together
in a pipeline that answers questions from web pages.</p></article></body></html>"""


class FakeStore:
    """Remembers which sources are stored, when they were loaded and what was stored for them."""

    def __init__(
        self,
        stored: set[str] | None = None,
        *,
        keeps_loaded_at: bool = False,
        keeps_versions: bool = False,
        records: dict[str, SourceRecord] | None = None,
    ) -> None:
        self.stored = stored or set()
        self.keeps_loaded_at = keeps_loaded_at
        self.keeps_versions = keeps_versions
        self.records = records or {}
        self.replaced: list[tuple[str, str, list[StoredChunk]]] = []
        self.written: list[SourceRecord | None] = []  # the record of each copy that was stored

    def has_source(self, url: str) -> bool:
        return url in self.stored

    def source_record(self, url: str) -> SourceRecord | None:
        return self.records.get(url, SourceRecord()) if url in self.stored else None

    def replace_source(
        self, url: str, title: str, chunks: Sequence[StoredChunk], record: SourceRecord | None = None
    ) -> None:
        self.replaced.append((url, title, list(chunks)))
        self.written.append(record)
        self.stored.add(url)
        if record is not None:
            self.records[url] = record


class FakeEmbedder:
    """Gives the n-th text the vector ``[n]`` and remembers the texts."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[float(number)] for number in range(len(texts))]


class FakeFetch:
    """Returns one document for every address and remembers the addresses."""

    def __init__(self, document: FetchedDocument | None = None, error: BaseException | None = None) -> None:
        self.document = document or FetchedDocument(kind="html", body=ARTICLE, url=URL)
        self.error = error
        self.urls: list[str] = []

    def __call__(self, url: str) -> FetchedDocument:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.document


def _ingestor(
    store: FakeStore,
    embedder: FakeEmbedder,
    fetch: FakeFetch,
    **options: Any,
) -> SourceIngestor:
    return SourceIngestor(store, embedder, fetch, pdf_max_pages=100, pdf_max_seconds=30.0, **options)


# --- a web page ---


def test_a_web_page_is_read_embedded_and_stored_with_its_title() -> None:
    store, embedder, fetch = FakeStore(), FakeEmbedder(), FakeFetch()

    result = _ingestor(store, embedder, fetch).ingest(URL)

    assert result == IngestResult(url=URL, reused=False, title="Sample Article", chunk_count=1)
    (url, title, chunks) = store.replaced[0]
    assert (url, title) == (URL, "Sample Article")
    assert "Retrieval systems find relevant passages" in chunks[0].text
    assert chunks[0].vector == [0.0]


def test_each_chunk_is_embedded_with_the_title_and_section_it_came_from() -> None:
    store, embedder, fetch = FakeStore(), FakeEmbedder(), FakeFetch()

    _ingestor(store, embedder, fetch).ingest(URL)

    (batch,) = embedder.batches
    assert batch[0].startswith("Sample Article\n\n")
    assert "Retrieval systems find relevant passages" in batch[0]


def test_a_long_page_is_split_into_chunks_that_keep_their_order() -> None:
    store, embedder, fetch = FakeStore(), FakeEmbedder(), FakeFetch()

    result = _ingestor(store, embedder, fetch, chunk_size=150, chunk_overlap=20).ingest(URL)

    chunks = store.replaced[0][2]
    assert result.chunk_count == len(chunks) > 1
    assert [chunk.vector for chunk in chunks] == [[float(number)] for number in range(len(chunks))]


def test_the_address_is_normalized_before_it_is_used() -> None:
    store, embedder, fetch = FakeStore(), FakeEmbedder(), FakeFetch()

    result = _ingestor(store, embedder, fetch).ingest(f"{URL}#section")

    assert result.url == URL
    assert fetch.urls == [URL]
    assert store.replaced[0][0] == URL


# --- a source that is already stored ---


def test_a_stored_source_is_reused_without_any_download_or_embedding() -> None:
    store, embedder, fetch = FakeStore({URL}), FakeEmbedder(), FakeFetch()

    result = _ingestor(store, embedder, fetch).ingest(URL)

    assert result == IngestResult(url=URL, reused=True)
    assert fetch.urls == []
    assert embedder.batches == []
    assert store.replaced == []


def test_a_refresh_downloads_a_stored_source_again() -> None:
    store, embedder, fetch = FakeStore({URL}), FakeEmbedder(), FakeFetch()

    result = _ingestor(store, embedder, fetch).ingest(URL, refresh=True)

    assert result.reused is False
    assert fetch.urls == [URL]
    assert len(store.replaced) == 1


# --- how old a stored source may be ---

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
WEEK = timedelta(days=7)


def _aged(
    age: timedelta | None, *, keeps_loaded_at: bool = True, max_age: timedelta | None = WEEK
) -> tuple[SourceIngestor, FakeStore, FakeFetch]:
    """An ingestor over a store that holds the source, loaded ``age`` ago (or at an unknown time)."""
    records = {URL: SourceRecord(loaded_at=None if age is None else NOW - age)}
    store, fetch = FakeStore({URL}, keeps_loaded_at=keeps_loaded_at, records=records), FakeFetch()
    return _ingestor(store, FakeEmbedder(), fetch, max_age=max_age, now=lambda: NOW), store, fetch


def test_a_stored_source_that_is_younger_than_the_limit_is_reused_and_its_age_is_reported() -> None:
    ingestor, store, fetch = _aged(timedelta(days=1))

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=True, age=timedelta(days=1))
    assert fetch.urls == []
    assert store.replaced == []


def test_a_stored_source_exactly_as_old_as_the_limit_is_still_reused() -> None:
    ingestor, _, fetch = _aged(WEEK)

    assert ingestor.ingest(URL).reused is True
    assert fetch.urls == []


def test_a_stored_source_older_than_the_limit_is_loaded_again() -> None:
    ingestor, store, fetch = _aged(timedelta(days=8))

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=False, title="Sample Article", chunk_count=1, expired=True)
    assert fetch.urls == [URL]
    assert len(store.replaced) == 1


def test_a_stored_source_of_an_unknown_age_is_loaded_again_once() -> None:
    ingestor, store, fetch = _aged(None)  # its chunks were stored before the store kept ages

    result = ingestor.ingest(URL)

    assert result.expired is True
    assert fetch.urls == [URL]
    assert len(store.replaced) == 1


def test_a_stored_source_is_kept_whatever_its_age_when_there_is_no_limit() -> None:
    ingestor, _, fetch = _aged(timedelta(days=100), max_age=None)

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=True, age=timedelta(days=100))
    assert fetch.urls == []


def test_the_limit_has_no_effect_when_the_store_cannot_tell_how_old_a_source_is() -> None:
    ingestor, store, fetch = _aged(None, keeps_loaded_at=False)  # a collection without the field

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=True)  # kept, and not loaded again on every question
    assert fetch.urls == []
    assert store.replaced == []


def test_a_refresh_loads_a_young_source_again_without_calling_it_expired() -> None:
    ingestor, _, fetch = _aged(timedelta(minutes=5))

    result = ingestor.ingest(URL, refresh=True)

    assert result.reused is False
    assert result.expired is False
    assert fetch.urls == [URL]


@pytest.mark.parametrize(("hours", "reloaded"), [(24, True), (0, False)])
def test_the_limit_comes_from_the_settings_and_zero_switches_it_off(
    monkeypatch: pytest.MonkeyPatch, hours: int, reloaded: bool
) -> None:
    fetched: list[str] = []

    def fake_fetch_document(url: str, **limits: float) -> FetchedDocument:
        fetched.append(url)
        return FetchedDocument(kind="html", body=ARTICLE, url=url)

    monkeypatch.setattr(ingest, "fetch_document", fake_fetch_document)
    for name, value in {
        "COHERE_API_KEY": "c",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "w",
        "GROQ_API_KEY": "g",
        "SOURCE_MAX_AGE_HOURS": str(hours),
    }.items():
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    old = SourceRecord(loaded_at=datetime.now(UTC) - timedelta(hours=25))
    store = FakeStore({URL}, keeps_loaded_at=True, records={URL: old})

    result = SourceIngestor.from_settings(settings, store, FakeEmbedder()).ingest(URL)

    assert result.reused is not reloaded  # a copy 25 hours old is too old for 24 hours, and fine without a limit
    assert bool(fetched) is reloaded


# --- asking the website whether the page changed ---

STORED_AT = NOW - timedelta(days=3)
ETAG = '"v1"'
MODIFIED = "Wed, 21 Oct 2026 07:28:00 GMT"
NEW_MODIFIED = "Thu, 22 Oct 2026 08:00:00 GMT"
TEXT_HASH = hashlib.sha256(clean_text(extract_page(ARTICLE, URL).text).encode("utf-8")).hexdigest()


def _record(
    content_hash: str | None = TEXT_HASH, etag: str | None = ETAG, last_modified: str | None = MODIFIED
) -> SourceRecord:
    return SourceRecord(loaded_at=STORED_AT, etag=etag, last_modified=last_modified, content_hash=content_hash)


class Clock:
    """A time that only moves when the test moves it."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class FakeRevalidate:
    """Answers the question of whether the page changed: with 'not modified', with a document or with a failure."""

    def __init__(self, answer: FetchedDocument | BaseException | None = None) -> None:
        self.answer = answer
        self.asked: list[tuple[str, Validators]] = []

    def __call__(self, url: str, marks: Validators) -> FetchedDocument | None:
        self.asked.append((url, marks))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _checking(
    record: SourceRecord | None = None,
    answer: FetchedDocument | BaseException | None = None,
    *,
    keeps_versions: bool = True,
    interval: timedelta = timedelta(0),
    clock: Callable[[], datetime] | None = None,
    max_age: timedelta | None = None,
    embedder: FakeEmbedder | None = None,
) -> tuple[SourceIngestor, FakeStore, FakeFetch, FakeRevalidate, FakeEmbedder]:
    """An ingestor over a store that holds the source, whose website answers ``answer`` to a question about a change."""
    store = FakeStore({URL}, keeps_loaded_at=True, keeps_versions=keeps_versions, records={URL: record or _record()})
    fetch, revalidate, embedder = FakeFetch(), FakeRevalidate(answer), embedder or FakeEmbedder()
    ingestor = _ingestor(
        store,
        embedder,
        fetch,
        revalidate=revalidate,
        check_interval=interval,
        now=clock or (lambda: NOW),
        max_age=max_age,
    )
    return ingestor, store, fetch, revalidate, embedder


def test_a_page_that_did_not_change_is_not_downloaded_and_its_stored_copy_is_reused() -> None:
    ingestor, store, fetch, revalidate, embedder = _checking()  # the website answers "not modified"

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=True, age=timedelta(days=3), checked=True)
    assert revalidate.asked == [(URL, Validators(etag=ETAG, last_modified=MODIFIED))]
    assert fetch.urls == []
    assert embedder.batches == []
    assert store.replaced == []


def test_a_page_that_is_sent_again_with_the_same_text_is_not_embedded_again() -> None:
    same = FetchedDocument(kind="html", body=ARTICLE, url=URL)  # a website that gives no version marks
    ingestor, store, _, _, embedder = _checking(answer=same)

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=True, age=timedelta(days=3), checked=True)
    assert embedder.batches == []
    assert store.replaced == []


def test_a_page_that_changed_is_loaded_again_with_its_new_version() -> None:
    newer = FetchedDocument(
        kind="html", body=ARTICLE, url=URL, validators=Validators(etag='"v2"', last_modified=NEW_MODIFIED)
    )
    ingestor, store, fetch, _, embedder = _checking(_record(content_hash="the fingerprint of the old text"), newer)

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=False, title="Sample Article", chunk_count=1, changed=True)
    assert fetch.urls == []  # the page that the check brought is used, and it is not downloaded a second time
    assert len(embedder.batches) == 1
    assert len(store.replaced) == 1
    assert store.written == [
        SourceRecord(
            loaded_at=NOW,
            etag='"v2"',
            last_modified=NEW_MODIFIED,
            content_hash=TEXT_HASH,
            links=(Link("https://example.com/", "Home"),),
        )
    ]


def test_the_links_of_a_page_are_stored_with_it() -> None:
    page = b"""<html><head><title>Guide</title></head><body>
<nav><a href="/pricing">Pricing</a> <a href="https://other.example/">Elsewhere</a></nav>
<article><h1>Guide</h1>
<p>Retrieval systems find relevant passages before a model writes an answer. This paragraph is long enough
for the extractor to treat it as real content rather than boilerplate on the page.</p>
<p>Read the <a href="/guide/details#start">details</a> for more about how chunking and embeddings work.</p>
</article></body></html>"""
    store, embedder = FakeStore(), FakeEmbedder()
    fetch = FakeFetch(FetchedDocument(kind="html", body=page, url=URL))

    _ingestor(store, embedder, fetch).ingest(URL)

    [record] = store.written
    assert record is not None
    assert record.links == (
        Link("https://example.com/pricing", "Pricing"),  # the menu counts, a link of another site does not
        Link("https://example.com/guide/details", "details"),
    )


def test_a_stored_copy_without_a_fingerprint_is_loaded_again_once_to_learn_its_version() -> None:
    record = _record(content_hash=None, etag=None, last_modified=None)  # stored before the versions were kept
    page = FetchedDocument(kind="html", body=ARTICLE, url=URL, validators=Validators(etag='"v1"'))
    ingestor, store, _, revalidate, _ = _checking(record, page)

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=False, title="Sample Article", chunk_count=1, expired=True)
    assert revalidate.asked == [(URL, Validators())]  # nothing was known, so it was an ordinary download
    assert store.written[0] is not None
    assert store.written[0].content_hash == TEXT_HASH


@pytest.mark.parametrize(
    "error",
    [
        FetchError("The website could not be reached."),
        InvalidURLError("That address is on a private network."),
        ExtractionError("That link is a file, not a web page."),
    ],
)
def test_when_the_page_cannot_be_checked_the_stored_copy_is_used_and_the_result_says_why(error: BaseException) -> None:
    ingestor, store, _, _, embedder = _checking(answer=error)

    result = ingestor.ingest(URL)

    assert result == IngestResult(url=URL, reused=True, age=timedelta(days=3), problem=str(error))
    assert store.replaced == []
    assert embedder.batches == []


def test_a_page_that_became_unreadable_keeps_the_stored_copy_and_says_why() -> None:
    empty = FetchedDocument(kind="html", body=b"<html><body><p>Hi</p></body></html>", url=URL)
    ingestor, store, _, _, _ = _checking(answer=empty)

    result = ingestor.ingest(URL)

    assert result.reused is True
    assert result.problem is not None
    assert "No readable text" in result.problem
    assert store.replaced == []


def test_when_the_new_version_cannot_be_embedded_the_earlier_copy_is_left_as_it_was() -> None:
    newer = FetchedDocument(kind="html", body=ARTICLE, url=URL)
    busy = FakeEmbedder(error=ServiceBusyError("Cohere is busy.", retry_after=30.0))
    ingestor, store, _, _, _ = _checking(_record(content_hash="the old text"), newer, embedder=busy)

    with pytest.raises(ServiceBusyError):
        ingestor.ingest(URL)

    assert store.replaced == []


def test_nothing_is_asked_when_checking_is_switched_off_for_the_call() -> None:
    ingestor, _, fetch, revalidate, _ = _checking()

    result = ingestor.ingest(URL, check=False)

    assert result == IngestResult(url=URL, reused=True, age=timedelta(days=3))
    assert revalidate.asked == []
    assert fetch.urls == []


def test_nothing_is_asked_when_the_store_does_not_keep_versions() -> None:
    ingestor, _, _, revalidate, _ = _checking(keeps_versions=False)  # a collection without the fields

    assert ingestor.ingest(URL) == IngestResult(url=URL, reused=True, age=timedelta(days=3))
    assert revalidate.asked == []


def test_nothing_is_asked_when_the_ingestor_has_no_way_to_ask() -> None:
    store = FakeStore({URL}, keeps_loaded_at=True, keeps_versions=True, records={URL: _record()})
    ingestor = _ingestor(store, FakeEmbedder(), FakeFetch(), now=lambda: NOW)

    assert ingestor.ingest(URL).reused is True


def test_a_refresh_downloads_the_page_without_asking_whether_it_changed() -> None:
    ingestor, _, fetch, revalidate, _ = _checking()

    result = ingestor.ingest(URL, refresh=True)

    assert result.reused is False
    assert revalidate.asked == []
    assert fetch.urls == [URL]


def test_a_page_that_is_too_old_is_loaded_again_without_asking_whether_it_changed() -> None:
    ingestor, _, fetch, revalidate, _ = _checking(max_age=timedelta(days=1))  # the copy is three days old

    result = ingestor.ingest(URL)

    assert result.expired is True
    assert revalidate.asked == []
    assert fetch.urls == [URL]


def test_a_page_is_not_asked_about_again_until_the_time_between_checks_has_passed() -> None:
    clock = Clock(NOW)
    ingestor, _, _, revalidate, _ = _checking(interval=timedelta(seconds=60), clock=clock)

    ingestor.ingest(URL)
    clock.now += timedelta(seconds=30)
    ingestor.ingest(URL)
    assert len(revalidate.asked) == 1  # the second question came within the minute

    clock.now += timedelta(seconds=31)
    ingestor.ingest(URL)
    assert len(revalidate.asked) == 2


def test_a_page_that_was_just_loaded_is_not_asked_about_within_the_time_between_checks() -> None:
    store = FakeStore(keeps_loaded_at=True, keeps_versions=True)  # nothing is stored yet
    revalidate = FakeRevalidate()
    ingestor = _ingestor(
        store,
        FakeEmbedder(),
        FakeFetch(),
        revalidate=revalidate,
        check_interval=timedelta(seconds=60),
        now=lambda: NOW,
    )

    first = ingestor.ingest(URL)
    second = ingestor.ingest(URL)

    assert first.reused is False
    assert second == IngestResult(url=URL, reused=True, age=timedelta(0))
    assert revalidate.asked == []


def test_the_check_and_its_limits_come_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Validators, dict[str, float]]] = []

    def fake_fetch_if_changed(url: str, marks: Validators, **limits: float) -> FetchedDocument | None:
        calls.append((url, marks, limits))
        return None

    monkeypatch.setattr(ingest, "fetch_if_changed", fake_fetch_if_changed)
    for name, value in {
        "COHERE_API_KEY": "c",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "w",
        "GROQ_API_KEY": "g",
        "SOURCE_CHECK_SECONDS": "3600",
        "FETCH_TIMEOUT_SECONDS": "7",
        "FETCH_MAX_BYTES": "1234",
        "PDF_MAX_BYTES": "5678",
    }.items():
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    store = FakeStore({URL}, keeps_loaded_at=True, keeps_versions=True, records={URL: _record()})
    ingestor = SourceIngestor.from_settings(settings, store, FakeEmbedder())

    first = ingestor.ingest(URL)
    second = ingestor.ingest(URL)  # within the hour, so the website is not asked again

    assert (first.checked, second.checked) == (True, False)
    assert calls == [
        (URL, Validators(etag=ETAG, last_modified=MODIFIED), {"timeout": 7.0, "max_bytes": 1234, "pdf_max_bytes": 5678})
    ]


# --- a PDF file ---


def test_a_pdf_file_is_read_page_by_page_and_stored_with_its_title() -> None:
    pdf = make_pdf([["First page of the manual."], ["Second page with the details."]], title="Manual")
    store, embedder = FakeStore(), FakeEmbedder()
    fetch = FakeFetch(FetchedDocument(kind="pdf", body=pdf, url="https://example.com/manual.pdf"))

    result = _ingestor(store, embedder, fetch).ingest("https://example.com/manual.pdf")

    assert result.title == "Manual"
    text = "\n".join(chunk.text for chunk in store.replaced[0][2])
    assert "First page of the manual." in text
    assert "Second page with the details." in text


def test_a_pdf_file_is_stored_without_links() -> None:
    pdf = make_pdf([["First page of the manual."]], title="Manual")
    store, embedder = FakeStore(), FakeEmbedder()
    fetch = FakeFetch(FetchedDocument(kind="pdf", body=pdf, url="https://example.com/manual.pdf"))

    _ingestor(store, embedder, fetch).ingest("https://example.com/manual.pdf")

    [record] = store.written
    assert record is not None
    assert record.links == ()


def test_the_page_limit_for_pdf_files_comes_from_the_ingestor() -> None:
    pdf = make_pdf([["First page of the manual."], ["Second page with the details."]], title="Manual")
    store, embedder = FakeStore(), FakeEmbedder()
    fetch = FakeFetch(FetchedDocument(kind="pdf", body=pdf, url="https://example.com/manual.pdf"))
    ingestor = SourceIngestor(store, embedder, fetch, pdf_max_pages=1, pdf_max_seconds=30.0)

    ingestor.ingest("https://example.com/manual.pdf")

    text = "\n".join(chunk.text for chunk in store.replaced[0][2])
    assert "Second page with the details." not in text
    assert "Only the first 1 of the 2 pages" in text


# --- what goes wrong ---


def test_a_failed_download_stores_nothing() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    fetch = FakeFetch(error=FetchError("The page could not be downloaded."))

    with pytest.raises(FetchError):
        _ingestor(store, embedder, fetch).ingest(URL)

    assert store.replaced == []


def test_a_page_without_readable_text_stores_nothing() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    fetch = FakeFetch(FetchedDocument(kind="html", body=b"<html><body><p>Hi</p></body></html>", url=URL))

    with pytest.raises(ExtractionError):
        _ingestor(store, embedder, fetch).ingest(URL)

    assert embedder.batches == []
    assert store.replaced == []


def test_text_that_makes_no_chunks_is_reported_as_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "chunk_text", lambda text, size, overlap: [])
    store, embedder, fetch = FakeStore(), FakeEmbedder(), FakeFetch()

    with pytest.raises(ExtractionError, match="No readable text"):
        _ingestor(store, embedder, fetch).ingest(URL)

    assert store.replaced == []


def test_when_embedding_fails_the_earlier_copy_is_left_as_it_was() -> None:
    store = FakeStore({URL})
    embedder = FakeEmbedder(error=ServiceBusyError("Cohere is busy.", retry_after=30.0))
    fetch = FakeFetch()

    with pytest.raises(ServiceBusyError):
        _ingestor(store, embedder, fetch).ingest(URL, refresh=True)

    assert store.replaced == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_sizes_that_make_no_sense_are_refused(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        _ingestor(FakeStore(), FakeEmbedder(), FakeFetch(), chunk_size=size, chunk_overlap=overlap)


# --- built from the settings ---


def test_the_limits_come_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, float]]] = []

    def fake_fetch_document(url: str, **limits: float) -> FetchedDocument:
        calls.append((url, limits))
        return FetchedDocument(kind="html", body=ARTICLE, url=url)

    monkeypatch.setattr(ingest, "fetch_document", fake_fetch_document)
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "w")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("FETCH_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("FETCH_MAX_BYTES", "1234")
    monkeypatch.setenv("PDF_MAX_BYTES", "5678")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    store, embedder = FakeStore(), FakeEmbedder()

    SourceIngestor.from_settings(settings, store, embedder).ingest(URL)

    assert calls == [(URL, {"timeout": 7.0, "max_bytes": 1234, "pdf_max_bytes": 5678})]
    assert len(store.replaced) == 1


# --- staying on a site ---


def _moved_to(url: str) -> FakeFetch:
    return FakeFetch(FetchedDocument(kind="html", body=ARTICLE, url=url))


def test_a_page_that_stays_on_its_site_is_stored() -> None:
    store, embedder = FakeStore(), FakeEmbedder()

    result = _ingestor(store, embedder, _moved_to("https://www.example.com/b")).ingest(URL, site="example.com")

    assert not result.reused
    assert len(store.replaced) == 1


def test_a_page_that_a_redirect_leads_to_another_site_is_refused_and_nothing_is_stored() -> None:
    store, embedder = FakeStore(), FakeEmbedder()

    with pytest.raises(FetchError) as caught:
        _ingestor(store, embedder, _moved_to("https://elsewhere.example/page")).ingest(URL, site="example.com")

    assert str(caught.value) == ingest.OFF_SITE_MESSAGE
    assert store.replaced == []
    assert embedder.batches == []


def test_any_site_is_allowed_when_none_is_named() -> None:
    store, embedder = FakeStore(), FakeEmbedder()

    _ingestor(store, embedder, _moved_to("https://elsewhere.example/page")).ingest(URL)

    assert len(store.replaced) == 1


def test_a_changed_page_that_now_leads_to_another_site_keeps_the_stored_copy() -> None:
    moved = FetchedDocument(kind="html", body=ARTICLE, url="https://elsewhere.example/page")
    ingestor, store, _, _, embedder = _checking(_record(content_hash="the fingerprint of the old text"), moved)

    result = ingestor.ingest(URL, site="example.com")

    assert result.reused
    assert result.problem == ingest.OFF_SITE_MESSAGE
    assert store.replaced == []
    assert embedder.batches == []
