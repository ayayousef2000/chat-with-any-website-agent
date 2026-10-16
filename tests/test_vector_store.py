"""Tests of the Weaviate store with fake Weaviate objects: no cluster is contacted."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import weaviate
from pydantic import SecretStr
from weaviate.classes.query import Filter, HybridFusion
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateConnectionError

from chat_with_any_website_agent.errors import ServiceBusyError, StoreError
from chat_with_any_website_agent.retrying import Retrier
from chat_with_any_website_agent.vector_store import (
    INSERT_BATCH_SIZE,
    NOT_MULTI_TENANT_MESSAGE,
    WEAVIATE_RETRYABLE,
    StoredChunk,
    WeaviateStore,
    open_tenant,
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
    def __init__(self, *, multi_tenant: bool = True, tenants: set[str] | None = None) -> None:
        self.config = SimpleNamespace(
            get=lambda: SimpleNamespace(multi_tenancy_config=SimpleNamespace(enabled=multi_tenant))
        )
        self.tenants = FakeTenants(set(tenants or ()))
        self.used_tenant: str | None = None

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
    assert [prop.name for prop in created["properties"]] == ["url", "title", "heading", "chunk_index", "text"]
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


def _store(client: FakeClient | None = None) -> tuple[WeaviateStore, FakeTenantCollection, list[float]]:
    pauses: list[float] = []
    retrier = Retrier(WEAVIATE_RETRYABLE, sleep=pauses.append, jitter=lambda _: 0.0)
    collection = FakeTenantCollection()
    fake: Any = collection
    return WeaviateStore(fake, client=client, retrier=retrier), collection, pauses


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
