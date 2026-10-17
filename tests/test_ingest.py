"""Tests of the ingestion of a source, with fakes instead of the network, Cohere and Weaviate."""

from typing import TYPE_CHECKING

import pytest

from chat_with_any_website_agent import ingest
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import ExtractionError, FetchError, ServiceBusyError
from chat_with_any_website_agent.fetching import FetchedDocument
from chat_with_any_website_agent.ingest import IngestResult, SourceIngestor
from tests.pdf_files import make_pdf

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """Remembers which sources are stored and what was stored for them."""

    def __init__(self, stored: set[str] | None = None) -> None:
        self.stored = stored or set()
        self.replaced: list[tuple[str, str, list[StoredChunk]]] = []

    def has_source(self, url: str) -> bool:
        return url in self.stored

    def replace_source(self, url: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        self.replaced.append((url, title, list(chunks)))


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
    **options: int,
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
