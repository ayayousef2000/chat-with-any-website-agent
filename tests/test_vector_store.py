"""Tests of the Weaviate store with fake Weaviate objects: no cluster is contacted."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import weaviate
from pydantic import SecretStr
from weaviate.classes.config import DataType
from weaviate.classes.query import Filter, HybridFusion
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateConnectionError

from chat_with_any_website_agent.errors import ServiceBusyError, StoreError
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.retrying import Retrier
from chat_with_any_website_agent.vector_store import (
    AGE_PROPERTY,
    ETAG_PROPERTY,
    HASH_PROPERTY,
    INSERT_BATCH_SIZE,
    LAST_MODIFIED_PROPERTY,
    LINKS_PROPERTY,
    NOT_MULTI_TENANT_MESSAGE,
    VERSION_PROPERTIES,
    WEAVIATE_RETRYABLE,
    SourceRecord,
    StoredChunk,
    WeaviateStore,
    ensure_record_properties,
    open_tenant,
    property_names,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _status_error(status: int) -> UnexpectedStatusCodeError:
    return UnexpectedStatusCodeError("Create collection", httpx.Response(status, text="refused"))


# --- fakes for opening the collection and the tenant ---


class FakeTenants:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.created: list[str] = []
        self.create_error: Exception | None = None
        self.someone_else_made_it = False  # the error is the answer to a race that another start won

    def exists(self, tenant: str) -> bool:
        return tenant in self.existing

    def create(self, tenant: str) -> None:
        self.created.append(tenant)
        if self.create_error is not None:
            if self.someone_else_made_it:
                self.existing.add(tenant)
            raise self.create_error
        self.existing.add(tenant)


class FakeCollection:
    def __init__(
        self, *, multi_tenant: bool = True, tenants: set[str] | None = None, properties: list[str] | None = None
    ) -> None:
        self.properties = list(properties or [])
        self.added: list[Any] = []
        self.config = SimpleNamespace(
            get=lambda: SimpleNamespace(
                multi_tenancy_config=SimpleNamespace(enabled=multi_tenant),
                properties=[SimpleNamespace(name=name) for name in self.properties],
            ),
            add_property=self._add_property,
        )
        self.tenants = FakeTenants(set(tenants or ()))
        self.used_tenant: str | None = None

    def _add_property(self, prop: Any) -> None:
        self.added.append(prop)
        self.properties.append(prop.name)

    def with_tenant(self, tenant: str) -> str:
        self.used_tenant = tenant
        return f"collection-for-{tenant}"


class FakeCollections:
    def __init__(self, existing: dict[str, FakeCollection] | None = None) -> None:
        self.existing = dict(existing or {})
        self.created: list[dict[str, Any]] = []
        self.create_error: Callable[[FakeCollections], Exception] | None = None
        self.new_collection = FakeCollection()

    def exists(self, name: str) -> bool:
        return name in self.existing

    def use(self, name: str) -> FakeCollection:
        return self.existing[name]

    def create(self, name: str, **kwargs: Any) -> FakeCollection:
        self.created.append({"name": name, **kwargs})
        if self.create_error is not None:
            raise self.create_error(self)
        self.existing[name] = self.new_collection
        return self.new_collection


class FakeClient:
    def __init__(self, collections: FakeCollections) -> None:
        self.collections = collections
        self.closed = False

    def close(self) -> None:
        self.closed = True


# --- opening the collection and the tenant ---


def test_a_new_collection_is_created_with_multi_tenancy_and_the_tenant_is_made() -> None:
    collections = FakeCollections()

    opened: Any = open_tenant(FakeClient(collections), "Chunks", "development")

    assert len(collections.created) == 1
    created = collections.created[0]
    assert created["name"] == "Chunks"
    assert created["multi_tenancy_config"].enabled is True
    assert [prop.name for prop in created["properties"]] == [
        "url",
        "title",
        "heading",
        "chunk_index",
        "text",
        "loaded_at",  # when the source was loaded
        "etag",  # the marks of the version of the page, and a fingerprint of its text, to notice a change
        "last_modified",
        "content_hash",
        "links",  # the links of the page that may be followed
    ]
    kinds = [prop.dataType for prop in created["properties"][-5:]]
    assert kinds == [DataType.DATE, DataType.TEXT, DataType.TEXT, DataType.TEXT, DataType.TEXT]
    assert collections.new_collection.tenants.created == ["development"]
    assert opened == "collection-for-development"  # every read and write goes through the tenant


def test_an_existing_multi_tenant_collection_is_reused_and_only_the_missing_tenant_is_made() -> None:
    existing = FakeCollection(tenants={"demo"})
    collections = FakeCollections({"Chunks": existing})

    opened: Any = open_tenant(FakeClient(collections), "Chunks", "development")

    assert collections.created == []  # the collection is not made again
    assert existing.tenants.created == ["development"]
    assert existing.tenants.existing == {"demo", "development"}  # the other tenant is left alone
    assert opened == "collection-for-development"


def test_a_tenant_that_already_exists_is_not_made_again() -> None:
    existing = FakeCollection(tenants={"demo"})

    open_tenant(FakeClient(FakeCollections({"Chunks": existing})), "Chunks", "demo")

    assert existing.tenants.created == []
    assert existing.used_tenant == "demo"


def test_three_environments_use_the_same_collection_with_their_own_tenants() -> None:
    collections = FakeCollections()
    client = FakeClient(collections)

    opened: list[Any] = [open_tenant(client, "Chunks", tenant) for tenant in ("development", "eval", "demo")]

    assert len(collections.created) == 1  # one collection, as the free plan allows
    assert opened == ["collection-for-development", "collection-for-eval", "collection-for-demo"]
    assert collections.new_collection.tenants.existing == {"development", "eval", "demo"}


def test_a_collection_without_multi_tenancy_is_refused_with_a_plain_message() -> None:
    client = FakeClient(FakeCollections({"Chunks": FakeCollection(multi_tenant=False)}))

    with pytest.raises(StoreError) as stopped:
        open_tenant(client, "Chunks", "development")

    assert str(stopped.value) == NOT_MULTI_TENANT_MESSAGE.format(name="Chunks")


def test_the_limit_of_one_collection_is_an_error() -> None:
    collections = FakeCollections()
    collections.create_error = lambda _: _status_error(429)

    with pytest.raises(UnexpectedStatusCodeError):
        open_tenant(FakeClient(collections), "Chunks", "development")


def test_a_collection_made_by_another_process_a_moment_ago_is_used() -> None:
    collections = FakeCollections()
    made_elsewhere = FakeCollection()

    def another_process_won(current: FakeCollections) -> Exception:
        current.existing["Chunks"] = made_elsewhere  # it exists by the time the error is looked at
        return _status_error(422)

    collections.create_error = another_process_won

    opened: Any = open_tenant(FakeClient(collections), "Chunks", "development")

    assert opened == "collection-for-development"
    assert made_elsewhere.tenants.created == ["development"]


def test_a_tenant_made_by_another_process_a_moment_ago_is_used() -> None:
    existing = FakeCollection()
    existing.tenants.create_error = _status_error(422)
    existing.tenants.someone_else_made_it = True

    opened: Any = open_tenant(FakeClient(FakeCollections({"Chunks": existing})), "Chunks", "development")

    assert opened == "collection-for-development"


def test_a_tenant_that_cannot_be_made_is_an_error() -> None:
    existing = FakeCollection()
    existing.tenants.create_error = _status_error(422)

    with pytest.raises(UnexpectedStatusCodeError):
        open_tenant(FakeClient(FakeCollections({"Chunks": existing})), "Chunks", "development")


# --- connecting ---


def test_connecting_opens_the_tenant_and_gives_the_client_the_startup_time(monkeypatch: pytest.MonkeyPatch) -> None:
    collections = FakeCollections()
    client = FakeClient(collections)
    captured: dict[str, Any] = {}

    def connect(**kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", connect)

    store = WeaviateStore.connect(
        "https://cluster.example", SecretStr("secret"), "Chunks", "development", init_timeout=25
    )

    assert captured["cluster_url"] == "https://cluster.example"
    assert captured["additional_config"].timeout.init == 25
    assert collections.new_collection.tenants.created == ["development"]
    store.close()
    assert client.closed is True


def _connect_to(monkeypatch: pytest.MonkeyPatch, collections: FakeCollections) -> WeaviateStore:
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", lambda **_: FakeClient(collections))
    return WeaviateStore.connect("https://cluster.example", SecretStr("secret"), "Chunks", "development")


def test_a_store_over_a_collection_with_all_the_fields_keeps_ages_versions_and_links(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fields = ["url", AGE_PROPERTY, *VERSION_PROPERTIES, LINKS_PROPERTY]
    collections = FakeCollections({"Chunks": FakeCollection(properties=fields)})

    with caplog.at_level(logging.WARNING, logger="chat_with_any_website_agent.vector_store"):
        store = _connect_to(monkeypatch, collections)

    assert (store.keeps_loaded_at, store.keeps_versions, store.keeps_links) == (True, True, True)
    assert caplog.text == ""  # nothing is missing, so there is nothing to warn about


@pytest.mark.parametrize(
    ("properties", "kept", "missing"),
    [
        (["url", "text"], (False, False, False), "loaded_at, etag, last_modified, content_hash, links"),
        (["url", AGE_PROPERTY], (True, False, False), "etag, last_modified, content_hash, links"),
        (["url", *VERSION_PROPERTIES, LINKS_PROPERTY], (False, True, True), "loaded_at"),
        # the versions are kept only when all three fields are there
        (
            ["url", AGE_PROPERTY, ETAG_PROPERTY, HASH_PROPERTY, LINKS_PROPERTY],
            (True, False, True),
            "last_modified",
        ),
        # a collection made before the links were kept
        (["url", AGE_PROPERTY, *VERSION_PROPERTIES], (True, True, False), "links"),
    ],
)
def test_a_store_over_a_collection_that_lacks_fields_says_which_and_how_to_add_them(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    properties: list[str],
    kept: tuple[bool, bool, bool],
    missing: str,
) -> None:
    collections = FakeCollections({"Chunks": FakeCollection(properties=properties)})

    with caplog.at_level(logging.WARNING, logger="chat_with_any_website_agent.vector_store"):
        store = _connect_to(monkeypatch, collections)

    assert (store.keeps_loaded_at, store.keeps_versions, store.keeps_links) == kept
    assert f"has no field {missing}," in caplog.text
    assert "Run 'agent-smoke --prepare' once to add them." in caplog.text
    assert collections.existing["Chunks"].added == []  # connecting never changes the collection


def test_a_failed_start_closes_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(FakeCollections({"Chunks": FakeCollection(multi_tenant=False)}))
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", lambda **_: client)

    with pytest.raises(StoreError):
        WeaviateStore.connect("https://cluster.example", SecretStr("secret"), "Chunks", "development")

    assert client.closed is True


# --- fakes for reading and writing ---


class FakeData:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.deleted: list[Any] = []
        self.inserted: list[list[Any]] = []
        self.failures: list[BaseException] = []
        self.insert_result: Any = SimpleNamespace(has_errors=False, errors={})

    def delete_many(self, where: Any) -> None:
        if self.failures:
            raise self.failures.pop(0)
        self.events.append("delete")
        self.deleted.append(where)

    def insert_many(self, objects: list[Any]) -> Any:
        self.events.append("insert")
        self.inserted.append(list(objects))
        return self.insert_result


class FakeQuery:
    def __init__(self) -> None:
        self.objects: list[Any] = []
        self.hybrid_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []

    def hybrid(self, **kwargs: Any) -> Any:
        self.hybrid_calls.append(kwargs)
        return SimpleNamespace(objects=self.objects)

    def fetch_objects(self, **kwargs: Any) -> Any:
        self.fetch_calls.append(kwargs)
        return SimpleNamespace(objects=self.objects)


class FakeAggregate:
    def __init__(self) -> None:
        self.groups: list[Any] = []
        self.total_count: int | None = None

    def over_all(self, **kwargs: Any) -> Any:
        return SimpleNamespace(groups=self.groups, total_count=self.total_count)


class FakeTenantCollection:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.data = FakeData(self.events)
        self.query = FakeQuery()
        self.aggregate = FakeAggregate()


def _store(
    client: FakeClient | None = None,
    *,
    keeps_loaded_at: bool = False,
    keeps_versions: bool = False,
    keeps_links: bool = False,
) -> tuple[WeaviateStore, FakeTenantCollection, list[float]]:
    pauses: list[float] = []
    retrier = Retrier(WEAVIATE_RETRYABLE, sleep=pauses.append, jitter=lambda _: 0.0)
    collection = FakeTenantCollection()
    fake: Any = collection
    store = WeaviateStore(
        fake,
        client=client,
        retrier=retrier,
        keeps_loaded_at=keeps_loaded_at,
        keeps_versions=keeps_versions,
        keeps_links=keeps_links,
    )
    return store, collection, pauses


def _found(url: str = "https://a.example/", heading: str | None = "Intro", score: float | None = 0.7) -> Any:
    properties = {"url": url, "title": "Page A", "heading": heading, "text": "some text", "chunk_index": 3}
    return SimpleNamespace(properties=properties, metadata=SimpleNamespace(score=score))


# --- storing ---


def test_the_old_chunks_of_a_source_are_replaced_by_the_new_ones() -> None:
    store, collection, _ = _store()
    chunks = [StoredChunk("first", "Intro", [0.1, 0.2]), StoredChunk("second", "", (0.3, 0.4))]

    store.replace_source("https://a.example/", "Page A", chunks)

    assert collection.events == ["delete", "insert"]  # the old chunks go before the new ones come
    assert collection.data.deleted == [Filter.by_property("url").equal("https://a.example/")]
    [inserted] = collection.data.inserted
    assert [item.properties for item in inserted] == [
        {"url": "https://a.example/", "title": "Page A", "heading": "Intro", "chunk_index": 0, "text": "first"},
        {"url": "https://a.example/", "title": "Page A", "heading": "", "chunk_index": 1, "text": "second"},
    ]
    assert [item.vector for item in inserted] == [[0.1, 0.2], [0.3, 0.4]]


def test_many_chunks_are_stored_in_batches() -> None:
    store, collection, _ = _store()
    chunks = [StoredChunk(f"text {n}", "", [0.0]) for n in range(250)]

    store.replace_source("https://a.example/", "Page A", chunks)

    assert [len(batch) for batch in collection.data.inserted] == [INSERT_BATCH_SIZE, INSERT_BATCH_SIZE, 50]


def test_a_source_without_chunks_is_only_cleared() -> None:
    store, collection, _ = _store()

    store.replace_source("https://a.example/", "Page A", [])

    assert collection.events == ["delete"]


def test_chunks_that_weaviate_refuses_are_an_error_with_a_count_and_the_first_reason() -> None:
    store, collection, _ = _store()
    collection.data.insert_result = SimpleNamespace(
        has_errors=True, errors={0: SimpleNamespace(message="too big"), 1: SimpleNamespace(message="other")}
    )

    chunks = [StoredChunk("a", "", [0.0]), StoredChunk("b", "", [0.0])]

    with pytest.raises(StoreError, match=r"Failed to store 2 chunk\(s\): too big"):
        store.replace_source("https://a.example/", "Page A", chunks)


def test_a_source_is_deleted_by_its_address() -> None:
    store, collection, _ = _store()

    store.delete_source("https://a.example/")

    assert collection.data.deleted == [Filter.by_property("url").equal("https://a.example/")]


def test_the_whole_tenant_can_be_cleared() -> None:
    store, collection, _ = _store()

    store.clear()

    assert collection.data.deleted == [Filter.by_property("chunk_index").greater_or_equal(0)]


# --- what is kept about the copy of a source ---

LOADED = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
SOURCE = "https://a.example/"
MODIFIED = "Wed, 21 Oct 2026 07:28:00 GMT"
MARKS = SourceRecord(loaded_at=LOADED, etag='"v1"', last_modified=MODIFIED, content_hash="abc123")
CHUNK = StoredChunk("a", "", [0.0])
# Weaviate's AND filter has no equality of its own, so the tests compare its parts.
FIRST_CHUNK_OF_SOURCE: Any = Filter.all_of(
    [Filter.by_property("url").equal(SOURCE), Filter.by_property("chunk_index").equal(0)]
)
LINKS = (Link("https://a.example/about", "About us"), Link("https://a.example/pricing", ""))
LINKS_JSON = '[["https://a.example/about","About us"],["https://a.example/pricing",""]]'


def test_a_source_is_stored_with_the_time_it_was_loaded_when_the_collection_keeps_ages() -> None:
    store, collection, _ = _store(keeps_loaded_at=True)
    chunks = [StoredChunk("a", "", [0.0]), StoredChunk("b", "", [0.0])]

    store.replace_source(SOURCE, "Page A", chunks, SourceRecord(loaded_at=LOADED))

    [inserted] = collection.data.inserted
    assert [item.properties[AGE_PROPERTY] for item in inserted] == [LOADED, LOADED]


def test_the_time_is_now_when_none_is_given() -> None:
    store, collection, _ = _store(keeps_loaded_at=True)
    before = datetime.now(UTC)

    store.replace_source(SOURCE, "Page A", [CHUNK])

    stored = collection.data.inserted[0][0].properties[AGE_PROPERTY]
    assert before <= stored <= datetime.now(UTC)


def test_nothing_about_the_copy_is_written_when_the_collection_lacks_the_fields() -> None:
    store, collection, _ = _store()  # the collection was made before the fields existed

    store.replace_source(SOURCE, "Page A", [CHUNK], MARKS)

    properties = collection.data.inserted[0][0].properties
    assert set(properties) == {"url", "title", "heading", "chunk_index", "text"}


def test_the_version_of_a_source_is_stored_with_its_chunks_when_the_collection_keeps_versions() -> None:
    store, collection, _ = _store(keeps_loaded_at=True, keeps_versions=True)

    store.replace_source(SOURCE, "Page A", [CHUNK], MARKS)

    properties = collection.data.inserted[0][0].properties
    assert properties[AGE_PROPERTY] == LOADED
    assert properties[ETAG_PROPERTY] == '"v1"'
    assert properties[LAST_MODIFIED_PROPERTY] == MODIFIED
    assert properties[HASH_PROPERTY] == "abc123"


def test_marks_that_the_website_did_not_give_are_not_written() -> None:
    store, collection, _ = _store(keeps_loaded_at=True, keeps_versions=True)

    store.replace_source(SOURCE, "Page A", [CHUNK], SourceRecord(loaded_at=LOADED, content_hash="abc123"))

    properties = collection.data.inserted[0][0].properties
    assert ETAG_PROPERTY not in properties
    assert LAST_MODIFIED_PROPERTY not in properties
    assert properties[HASH_PROPERTY] == "abc123"


def test_the_version_is_not_written_when_only_the_age_field_exists() -> None:
    store, collection, _ = _store(keeps_loaded_at=True)  # a collection upgraded before the version fields existed

    store.replace_source(SOURCE, "Page A", [CHUNK], MARKS)

    properties = collection.data.inserted[0][0].properties
    assert properties[AGE_PROPERTY] == LOADED
    assert not set(VERSION_PROPERTIES) & set(properties)


def test_the_links_of_a_page_are_stored_on_its_first_chunk_only() -> None:
    store, collection, _ = _store(keeps_links=True)
    chunks = [StoredChunk("a", "", [0.0]), StoredChunk("b", "", [0.0])]

    store.replace_source(SOURCE, "Page A", chunks, SourceRecord(links=LINKS))

    first, second = collection.data.inserted[0]
    assert first.properties[LINKS_PROPERTY] == LINKS_JSON
    assert LINKS_PROPERTY not in second.properties


def test_a_page_without_links_stores_no_links_field() -> None:
    store, collection, _ = _store(keeps_links=True)

    store.replace_source(SOURCE, "Page A", [CHUNK], SourceRecord())

    assert LINKS_PROPERTY not in collection.data.inserted[0][0].properties


def test_links_are_not_written_when_the_collection_lacks_the_field() -> None:
    store, collection, _ = _store()  # a collection made before the links were kept

    store.replace_source(SOURCE, "Page A", [CHUNK], SourceRecord(links=LINKS))

    assert LINKS_PROPERTY not in collection.data.inserted[0][0].properties


def test_the_links_of_a_page_are_read_back_from_its_first_chunk() -> None:
    store, collection, _ = _store(keeps_links=True)
    collection.query.objects = [SimpleNamespace(properties={LINKS_PROPERTY: LINKS_JSON})]

    record = store.source_record(SOURCE)

    assert record is not None
    assert record.links == LINKS
    assert collection.query.fetch_calls[0]["filters"].filters == FIRST_CHUNK_OF_SOURCE.filters


@pytest.mark.parametrize("stored", [None, "", "not json", '{"a": 1}', "[1, 2]"])
def test_links_that_are_missing_or_unreadable_are_no_links(stored: object) -> None:
    store, collection, _ = _store(keeps_links=True)
    collection.query.objects = [SimpleNamespace(properties={LINKS_PROPERTY: stored})]

    record = store.source_record(SOURCE)

    assert record is not None
    assert record.links == ()


def test_the_record_of_a_source_is_read_from_one_of_its_chunks() -> None:
    store, collection, _ = _store(keeps_loaded_at=True, keeps_versions=True)
    collection.query.objects = [
        SimpleNamespace(
            properties={
                AGE_PROPERTY: LOADED,
                ETAG_PROPERTY: '"v1"',
                LAST_MODIFIED_PROPERTY: MODIFIED,
                HASH_PROPERTY: "abc123",
            }
        )
    ]

    assert store.source_record(SOURCE) == MARKS
    [call] = collection.query.fetch_calls
    assert call["filters"].filters == FIRST_CHUNK_OF_SOURCE.filters  # an AND of the address and the first chunk
    assert (call["limit"], call["return_properties"]) == (1, [AGE_PROPERTY, *VERSION_PROPERTIES])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (LOADED, LOADED),
        ("2026-10-01T12:00:00Z", LOADED),
        ("2026-10-01T12:00:00+00:00", LOADED),
        (LOADED.replace(tzinfo=None), LOADED),  # a time without a time zone is taken as UTC
        (None, None),  # the chunks were stored before the field existed
        ("not a date", None),
        (12345, None),
    ],
)
def test_a_time_is_read_whatever_form_weaviate_returns_it_in(value: object, expected: datetime | None) -> None:
    store, collection, _ = _store(keeps_loaded_at=True)
    collection.query.objects = [SimpleNamespace(properties={AGE_PROPERTY: value})]

    record = store.source_record(SOURCE)

    assert record is not None
    assert record.loaded_at == expected


def test_a_copy_stored_before_the_fields_existed_has_an_empty_record() -> None:
    store, collection, _ = _store(keeps_loaded_at=True, keeps_versions=True)
    collection.query.objects = [SimpleNamespace(properties={})]

    assert store.source_record(SOURCE) == SourceRecord()


def test_marks_that_are_empty_or_not_text_are_not_known() -> None:
    store, collection, _ = _store(keeps_loaded_at=True, keeps_versions=True)
    collection.query.objects = [
        SimpleNamespace(properties={ETAG_PROPERTY: "", LAST_MODIFIED_PROPERTY: 5, HASH_PROPERTY: None})
    ]

    assert store.source_record(SOURCE) == SourceRecord()


def _section_chunk(heading: str, index: int, text: str = "text of the section") -> Any:
    properties = {"url": SOURCE, "title": "Paper", "heading": heading, "text": text, "chunk_index": index}
    return SimpleNamespace(properties=properties, metadata=SimpleNamespace(score=None))


def test_a_section_is_read_by_its_heading_in_the_order_of_the_source() -> None:
    store, collection, _ = _store()
    collection.query.objects = [_section_chunk("Page 5", 7, "first part"), _section_chunk("Page 5", 8, "second part")]

    chunks = store.fetch_section(SOURCE, "Page 5", 20)

    assert [(chunk.chunk_index, chunk.text) for chunk in chunks] == [(7, "first part"), (8, "second part")]
    assert all(chunk.url == SOURCE and chunk.title == "Paper" and chunk.score is None for chunk in chunks)
    [call] = collection.query.fetch_calls
    expected: Any = Filter.all_of(
        [Filter.by_property("url").equal(SOURCE), Filter.by_property("heading").equal("Page 5")]
    )
    assert call["filters"].filters == expected.filters
    assert call["limit"] == 20
    assert [(sort.prop, sort.ascending) for sort in call["sort"].sorts] == [("chunk_index", True)]
    assert call["return_properties"] == ["url", "title", "heading", "text", "chunk_index"]


def test_chunks_are_read_by_their_places_in_the_order_of_the_source() -> None:
    store, collection, _ = _store()
    collection.query.objects = [_section_chunk("Intro", 4, "before"), _section_chunk("Intro", 6, "after")]

    chunks = store.fetch_chunks(SOURCE, [6, 4, 6])

    assert [(chunk.chunk_index, chunk.text) for chunk in chunks] == [(4, "before"), (6, "after")]
    assert all(chunk.url == SOURCE and chunk.title == "Paper" and chunk.score is None for chunk in chunks)
    [call] = collection.query.fetch_calls
    expected: Any = Filter.all_of(
        [Filter.by_property("url").equal(SOURCE), Filter.by_property("chunk_index").contains_any([4, 6])]
    )
    assert call["filters"].filters == expected.filters  # each place once, in order
    assert call["limit"] == 2
    assert [(sort.prop, sort.ascending) for sort in call["sort"].sorts] == [("chunk_index", True)]
    assert call["return_properties"] == ["url", "title", "heading", "text", "chunk_index"]


def test_a_chunk_that_was_not_asked_for_or_is_of_another_source_is_left_out() -> None:
    store, collection, _ = _store()
    other = _section_chunk("Intro", 4)
    other.properties["url"] = "https://other.example/"
    collection.query.objects = [other, _section_chunk("Intro", 5), _section_chunk("Intro", 6)]

    chunks = store.fetch_chunks(SOURCE, [6])

    assert [chunk.chunk_index for chunk in chunks] == [6]


def test_no_places_are_asked_for_without_a_request_to_weaviate() -> None:
    store, collection, _ = _store()

    assert store.fetch_chunks(SOURCE, []) == []
    assert collection.query.fetch_calls == []


def test_a_place_with_no_chunk_gives_no_chunk() -> None:
    store, _, _ = _store()

    assert store.fetch_chunks(SOURCE, [99]) == []


def test_a_section_with_a_similar_heading_is_left_out() -> None:
    # Weaviate matches the words of a heading, so "Page 5 notes" or "5 Page" can come back with "Page 5".
    store, collection, _ = _store()
    collection.query.objects = [
        _section_chunk("Page 5", 7),
        _section_chunk("Page 5 notes", 9),
        _section_chunk("5 Page", 10),
    ]

    chunks = store.fetch_section(SOURCE, "Page 5", 20)

    assert [chunk.heading for chunk in chunks] == ["Page 5"]


def test_a_source_without_the_section_gives_no_chunks() -> None:
    store, _, _ = _store()

    assert store.fetch_section(SOURCE, "Page 5", 20) == []


def test_a_chunk_without_a_heading_is_read_with_an_empty_heading_and_left_out_of_a_section() -> None:
    store, collection, _ = _store()
    collection.query.objects = [_section_chunk(None, 1)]  # type: ignore[arg-type]

    assert store.fetch_section(SOURCE, "Page 5", 20) == []


def test_the_title_of_a_source_is_read_from_its_first_chunk() -> None:
    store, collection, _ = _store()
    collection.query.objects = [SimpleNamespace(properties={"title": "Control Flow"})]

    assert store.source_title(SOURCE) == "Control Flow"
    [call] = collection.query.fetch_calls
    assert call["filters"].filters == FIRST_CHUNK_OF_SOURCE.filters
    assert (call["limit"], call["return_properties"]) == (1, ["title"])


@pytest.mark.parametrize("stored", [None, "", 5])
def test_a_source_whose_title_is_missing_has_no_title(stored: object) -> None:
    store, collection, _ = _store()
    collection.query.objects = [SimpleNamespace(properties={"title": stored})]

    assert store.source_title(SOURCE) is None


def test_the_title_of_a_source_that_is_not_stored_is_none() -> None:
    store, _, _ = _store()

    assert store.source_title(SOURCE) is None


def test_the_record_of_a_source_that_is_not_stored_is_none() -> None:
    store, _, _ = _store(keeps_loaded_at=True, keeps_versions=True)

    assert store.source_record(SOURCE) is None


@pytest.mark.parametrize(
    ("keeps_loaded_at", "keeps_versions", "keeps_links", "asked"),
    [
        (True, True, True, [AGE_PROPERTY, *VERSION_PROPERTIES, LINKS_PROPERTY]),
        (True, False, False, [AGE_PROPERTY]),
        (False, True, False, list(VERSION_PROPERTIES)),
        (False, False, True, [LINKS_PROPERTY]),
        (False, False, False, []),
    ],
)
def test_only_the_fields_that_the_collection_has_are_asked_for(
    keeps_loaded_at: bool, keeps_versions: bool, keeps_links: bool, asked: list[str]
) -> None:
    store, collection, _ = _store(
        keeps_loaded_at=keeps_loaded_at, keeps_versions=keeps_versions, keeps_links=keeps_links
    )
    collection.query.objects = [SimpleNamespace(properties={})]

    store.source_record(SOURCE)

    assert collection.query.fetch_calls[0]["return_properties"] == asked


def test_the_fields_are_added_to_a_collection_that_lacks_them() -> None:
    collection = FakeCollection(properties=["url", "text"])

    added = ensure_record_properties(collection)

    assert added == [AGE_PROPERTY, *VERSION_PROPERTIES, LINKS_PROPERTY]
    types = {prop.name: prop.dataType for prop in collection.added}
    assert types == {
        AGE_PROPERTY: DataType.DATE,
        ETAG_PROPERTY: DataType.TEXT,
        LAST_MODIFIED_PROPERTY: DataType.TEXT,
        HASH_PROPERTY: DataType.TEXT,
        LINKS_PROPERTY: DataType.TEXT,
    }
    assert property_names(collection) == {"url", "text", AGE_PROPERTY, *VERSION_PROPERTIES, LINKS_PROPERTY}


def test_only_the_missing_fields_are_added() -> None:
    collection = FakeCollection(properties=["url", AGE_PROPERTY])  # upgraded before the version fields existed
    expected = [*VERSION_PROPERTIES, LINKS_PROPERTY]

    assert ensure_record_properties(collection) == expected
    assert [prop.name for prop in collection.added] == expected


def test_nothing_is_added_when_the_collection_has_all_the_fields() -> None:
    collection = FakeCollection(properties=["url", AGE_PROPERTY, *VERSION_PROPERTIES, LINKS_PROPERTY])

    assert ensure_record_properties(collection) == []
    assert collection.added == []


# --- looking things up ---


def test_the_sources_are_listed_with_their_number_of_chunks() -> None:
    store, collection, _ = _store()
    collection.aggregate.groups = [
        SimpleNamespace(grouped_by=SimpleNamespace(value="https://a.example/"), total_count=3),
        SimpleNamespace(grouped_by=SimpleNamespace(value="https://b.example/"), total_count=None),
    ]

    assert store.list_sources() == {"https://a.example/": 3, "https://b.example/": 0}


@pytest.mark.parametrize(("total", "expected"), [(42, 42), (None, 0)])
def test_the_chunks_are_counted(total: int | None, expected: int) -> None:
    store, collection, _ = _store()
    collection.aggregate.total_count = total

    assert store.count_chunks() == expected


@pytest.mark.parametrize(("objects", "expected"), [([object()], True), ([], False)])
def test_a_source_is_known_when_it_has_a_chunk(objects: list[Any], expected: bool) -> None:
    store, collection, _ = _store()
    collection.query.objects = objects

    assert store.has_source("https://a.example/") is expected
    assert collection.query.fetch_calls[0]["filters"] == Filter.by_property("url").equal("https://a.example/")
    assert collection.query.fetch_calls[0]["limit"] == 1


# --- searching ---


def test_a_search_is_a_hybrid_search_and_the_chunks_carry_their_source() -> None:
    store, collection, _ = _store()
    collection.query.objects = [_found(), _found(url="https://b.example/", heading=None, score=None)]

    result = store.search("what is it?", (0.1, 0.2), limit=5, alpha=0.65)

    call = collection.query.hybrid_calls[0]
    assert call["query"] == "what is it?"
    assert call["vector"] == [0.1, 0.2]
    assert (call["alpha"], call["limit"]) == (0.65, 5)
    assert call["fusion_type"] == HybridFusion.RELATIVE_SCORE
    assert call["query_properties"] == ["text", "heading"]
    assert "url" in call["return_properties"]
    assert call["filters"] is None  # no sources named: all of them are searched
    assert [(chunk.url, chunk.heading, chunk.score, chunk.chunk_index) for chunk in result] == [
        ("https://a.example/", "Intro", 0.7, 3),
        ("https://b.example/", "", None, 3),
    ]
    assert result[0].title == "Page A"
    assert result[0].text == "some text"


def test_a_search_in_one_source_filters_on_its_address() -> None:
    store, collection, _ = _store()

    store.search("q", [0.0], limit=3, alpha=0.5, urls=["https://a.example/"])

    assert collection.query.hybrid_calls[0]["filters"] == Filter.by_property("url").equal("https://a.example/")


def test_a_search_in_several_sources_filters_on_any_of_them() -> None:
    store, collection, _ = _store()
    urls = ["https://a.example/", "https://b.example/"]

    store.search("q", [0.0], limit=3, alpha=0.5, urls=urls)

    assert collection.query.hybrid_calls[0]["filters"] == Filter.by_property("url").contains_any(urls)


def test_a_search_in_no_sources_finds_nothing_without_asking_weaviate() -> None:
    store, collection, _ = _store()

    assert store.search("q", [0.0], limit=3, alpha=0.5, urls=[]) == []
    assert collection.query.hybrid_calls == []


# --- failures and closing ---


def test_a_connection_that_fails_is_tried_again() -> None:
    store, collection, pauses = _store()
    collection.data.failures = [WeaviateConnectionError("down")]

    store.delete_source("https://a.example/")

    assert len(collection.data.deleted) == 1
    assert pauses == [1.0]


def test_when_weaviate_stays_unreachable_the_caller_gets_a_busy_error() -> None:
    store, collection, _ = _store()
    collection.data.failures = [WeaviateConnectionError("down")] * 4

    with pytest.raises(ServiceBusyError) as raised:
        store.delete_source("https://a.example/")

    assert str(raised.value) == "Weaviate is busy right now. Try again in a few seconds."


def test_an_error_that_retrying_cannot_fix_reaches_the_caller_at_once() -> None:
    store, collection, pauses = _store()
    collection.data.failures = [ValueError("a bug")]

    with pytest.raises(ValueError, match="a bug"):
        store.delete_source("https://a.example/")

    assert pauses == []


def test_closing_closes_the_connection_the_store_owns_and_a_with_block_does_too() -> None:
    client = FakeClient(FakeCollections())
    store, _, _ = _store(client)

    with store:
        assert client.closed is False

    assert client.closed is True


def test_closing_a_store_that_does_not_own_the_connection_is_harmless() -> None:
    store, _, _ = _store()

    store.close()
