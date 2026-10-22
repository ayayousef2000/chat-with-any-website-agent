"""Storing chunk vectors in Weaviate and searching them with a hybrid (keyword and vector) search.

Everything is kept in one multi-tenant collection with one tenant for each environment (for example ``development``,
``eval`` and ``demo``). Weaviate keeps the tenants apart, so one environment never sees or deletes the chunks of
another, even though a free Weaviate Cloud plan allows only one collection.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

import weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery, Sort
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateConnectionError, WeaviateTimeoutError

from chat_with_any_website_agent.errors import StoreError
from chat_with_any_website_agent.links import Link, links_from_json, links_to_json
from chat_with_any_website_agent.retrying import Retrier

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from pydantic import SecretStr
    from weaviate.collections import Collection

NOT_MULTI_TENANT_MESSAGE = (
    "The Weaviate collection {name} exists but is not multi-tenant, so it cannot keep the chunks of several "
    "environments apart. Delete the collection in the Weaviate console and run again: it is created again with "
    "multi-tenancy, and the chunks in it are only copies of pages."
)

# A connection that failed or timed out is worth trying again; the other errors would fail the same way again.
WEAVIATE_RETRYABLE: tuple[type[BaseException], ...] = (WeaviateConnectionError, WeaviateTimeoutError)

INSERT_BATCH_SIZE = 100

# The fields of every chunk that describe the copy of its source: when it was loaded, and which version of the page it
# is (the marks the website gave, and a fingerprint of the cleaned text), so that a changed page can be noticed.
AGE_PROPERTY = "loaded_at"
ETAG_PROPERTY = "etag"
LAST_MODIFIED_PROPERTY = "last_modified"
HASH_PROPERTY = "content_hash"
VERSION_PROPERTIES = (ETAG_PROPERTY, LAST_MODIFIED_PROPERTY, HASH_PROPERTY)
# The links of the page that may be followed, as a short JSON text. They are kept on the first chunk of the page only.
LINKS_PROPERTY = "links"
FIRST_CHUNK = 0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRecord:
    """What is kept about the stored copy of a source, next to its chunks.

    Attributes:
        loaded_at: When the copy was loaded.
        etag: The ``ETag`` that the website gave with the page.
        last_modified: The ``Last-Modified`` date that the website gave with the page.
        content_hash: A fingerprint of the cleaned text of the page, to tell whether the text changed.
        links: The links of the page that may be followed, in the order of the page.
    """

    loaded_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None
    links: tuple[Link, ...] = ()


@dataclass(frozen=True)
class StoredChunk:
    """A chunk and its embedding, ready to be stored.

    Attributes:
        text: The text of the chunk.
        heading: The heading of the section the chunk belongs to.
        vector: The embedding of the chunk.
    """

    text: str
    heading: str
    vector: Sequence[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a search.

    Attributes:
        url: The address of the source the chunk came from.
        title: The title of the source.
        heading: The heading of the section the chunk belongs to, or an empty string.
        text: The text of the chunk.
        chunk_index: The position of the chunk in its source.
        score: The relevance score of the search, when there is one.
    """

    url: str
    title: str
    heading: str
    text: str
    chunk_index: int
    score: float | None


def open_tenant(client: Any, name: str, tenant: str) -> Collection[Any, Any]:
    """Open the collection for one tenant, creating the collection and the tenant if they are missing.

    Args:
        client: A connected Weaviate client.
        name: The name of the collection.
        tenant: The tenant of the environment.

    Returns:
        The collection, bound to the tenant, so that every read and write goes through it.

    Raises:
        StoreError: If the collection exists but is not multi-tenant.
    """
    collection = _get_or_create_collection(client, name)
    if not collection.config.get().multi_tenancy_config.enabled:
        raise StoreError(NOT_MULTI_TENANT_MESSAGE.format(name=name))
    if not collection.tenants.exists(tenant):
        try:
            collection.tenants.create(tenant)
        except UnexpectedStatusCodeError:
            if not collection.tenants.exists(tenant):  # the same tenant may have been created a moment ago
                raise
    return collection.with_tenant(tenant)


def _existing_collection(client: Any, name: str) -> Collection[Any, Any]:
    collection: Collection[Any, Any] = client.collections.use(name)
    return collection


def _record_properties() -> list[Property]:
    """The fields that describe the copy of a source. They are not searched, only read."""
    return [
        Property(name=AGE_PROPERTY, data_type=DataType.DATE),
        *(
            Property(name=name, data_type=DataType.TEXT, index_searchable=False)
            for name in (*VERSION_PROPERTIES, LINKS_PROPERTY)
        ),
    ]


def _get_or_create_collection(client: Any, name: str) -> Collection[Any, Any]:
    if client.collections.exists(name):
        return _existing_collection(client, name)
    try:
        created: Collection[Any, Any] = client.collections.create(
            name,
            properties=[
                # FIELD tokenization keeps the whole URL as one token so filters match it exactly.
                Property(name="url", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="title", data_type=DataType.TEXT, index_searchable=False),
                Property(name="heading", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
                *_record_properties(),
            ],
            # Vectors are computed by the application (Cohere), not by Weaviate.
            vector_config=Configure.Vectors.self_provided(),
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
        )
    except UnexpectedStatusCodeError:
        # Another process may have created the collection while this one was starting.
        if client.collections.exists(name):
            return _existing_collection(client, name)
        raise
    return created


def property_names(collection: Any) -> set[str]:
    """The names of the properties of a collection."""
    return {prop.name for prop in collection.config.get().properties}


def ensure_record_properties(collection: Any) -> list[str]:
    """Add the fields that describe the copy of a source, for the ones the collection does not have yet.

    The change only adds fields. No chunk is changed or deleted, and no embedding is made. The chunks that were
    stored before have no value in the fields, so the age and the version of their copy are not known.

    Args:
        collection: The collection, not bound to a tenant.

    Returns:
        The names of the fields that were added. Empty when the collection had them all.
    """
    existing = property_names(collection)
    added = []
    for prop in _record_properties():
        if prop.name not in existing:
            collection.config.add_property(prop)
            added.append(prop.name)
    return added


def _as_datetime(value: Any) -> datetime | None:
    """Read the date that Weaviate returns, which is a ``datetime`` or a text, as a time with its time zone."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _as_text(value: Any) -> str | None:
    """Read a text field that may be missing, so that a missing or empty value is ``None``."""
    return value if isinstance(value, str) and value else None


def _retrieved_chunk(properties: Any, score: float | None) -> RetrievedChunk:
    """A chunk from the properties that Weaviate returns for an object."""
    return RetrievedChunk(
        url=str(properties["url"]),
        title=str(properties["title"]),
        heading=str(properties.get("heading") or ""),
        text=str(properties["text"]),
        chunk_index=int(properties["chunk_index"]),
        score=score,
    )


def _url_filter(url: str) -> Any:
    return Filter.by_property("url").equal(url)


class WeaviateStore:
    """Stores chunk vectors in the tenant of one environment and searches them."""

    def __init__(
        self,
        collection: Collection[Any, Any],
        *,
        client: Any = None,
        retrier: Retrier | None = None,
        keeps_loaded_at: bool = False,
        keeps_versions: bool = False,
        keeps_links: bool = False,
    ) -> None:
        """Create a store over a collection that is already bound to a tenant.

        Args:
            collection: The collection for the tenant, as ``open_tenant`` returns it.
            client: The connected client, closed by ``close``. Not needed when something else closes it.
            retrier: How to try again after a failed connection. The default suits a person who is waiting.
            keeps_loaded_at: Whether the collection has the field that says when a source was loaded. When it does
                not, nothing is written to that field and the age of a source is not known.
            keeps_versions: Whether the collection has the fields that say which version of a page a copy is. When it
                does not, nothing is written to them and a changed page cannot be noticed.
            keeps_links: Whether the collection has the field for the links of a page. When it does not, the links
                are not kept and the agent cannot follow them.
        """
        self._collection = collection
        self._client = client
        self._retrier = retrier or Retrier(retryable=WEAVIATE_RETRYABLE)
        self._keeps_loaded_at = keeps_loaded_at
        self._keeps_versions = keeps_versions
        self._keeps_links = keeps_links

    @property
    def keeps_links(self) -> bool:
        """Whether the collection keeps the links of each page."""
        return self._keeps_links

    @property
    def keeps_loaded_at(self) -> bool:
        """Whether the collection keeps when each source was loaded."""
        return self._keeps_loaded_at

    @property
    def keeps_versions(self) -> bool:
        """Whether the collection keeps which version of a page each copy is."""
        return self._keeps_versions

    @classmethod
    def connect(
        cls,
        url: str,
        api_key: SecretStr,
        collection_name: str,
        tenant: str,
        *,
        init_timeout: int = 30,
        retrier: Retrier | None = None,
    ) -> Self:
        """Connect to a Weaviate Cloud cluster and open the tenant of an environment.

        Args:
            url: The cluster address.
            api_key: The Weaviate key.
            collection_name: The name of the collection.
            tenant: The tenant of the environment.
            init_timeout: The seconds the client allows for its startup checks, which a slow connection can miss
                with the 2 seconds it allows by default.
            retrier: How to try again after a failed connection.

        Returns:
            The store, connected and ready.

        Raises:
            StoreError: If the collection exists but is not multi-tenant.
        """
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key.get_secret_value()),
            additional_config=AdditionalConfig(timeout=Timeout(init=init_timeout)),
        )
        try:
            collection = open_tenant(client, collection_name, tenant)
            names = property_names(client.collections.use(collection_name))
        except Exception:
            client.close()  # a failed start must not leave the connection open
            raise
        missing = [prop.name for prop in _record_properties() if prop.name not in names]
        if missing:
            logger.warning(
                "The collection %s has no field %s, so the age, the version and the links of the stored sources are "
                "not known (a page that changed is not noticed and no link is followed). "
                "Run 'agent-smoke --prepare' once to add them.",
                collection_name,
                ", ".join(missing),
            )
        return cls(
            collection,
            client=client,
            retrier=retrier,
            keeps_loaded_at=AGE_PROPERTY in names,
            keeps_versions=all(name in names for name in VERSION_PROPERTIES),
            keeps_links=LINKS_PROPERTY in names,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def replace_source(
        self, url: str, title: str, chunks: Sequence[StoredChunk], record: SourceRecord | None = None
    ) -> None:
        """Remove any chunks previously stored for a source and store the new ones.

        Args:
            url: The address of the source.
            title: The title of the source.
            chunks: The chunks with their embeddings, in reading order.
            record: What is kept about this copy of the source. The time it was loaded is now when it is not given.
                A field is only written when the collection has it. The links are kept on the first chunk only.

        Raises:
            StoreError: If Weaviate refuses to store some of the chunks.
            ServiceBusyError: If Weaviate cannot be reached after the retries.
        """
        record = record or SourceRecord()
        age = self._record_values(record)
        links = {LINKS_PROPERTY: links_to_json(record.links)} if self._keeps_links and record.links else {}
        objects = [
            DataObject(
                properties={
                    "url": url,
                    "title": title,
                    "heading": chunk.heading,
                    "chunk_index": index,
                    "text": chunk.text,
                    **age,
                    **(links if index == FIRST_CHUNK else {}),
                },
                vector=list(chunk.vector),
            )
            for index, chunk in enumerate(chunks)
        ]

        def replace() -> None:
            self._collection.data.delete_many(where=_url_filter(url))
            for start in range(0, len(objects), INSERT_BATCH_SIZE):
                result = self._collection.data.insert_many(objects[start : start + INSERT_BATCH_SIZE])
                if result.has_errors:
                    first_error = next(iter(result.errors.values()))
                    raise StoreError(f"Failed to store {len(result.errors)} chunk(s): {first_error.message}")

        self._retrier.call(replace, name="Weaviate")

    def delete_source(self, url: str) -> None:
        """Delete every chunk stored for a source.

        Args:
            url: The address of the source.
        """
        self._retrier.call(lambda: self._collection.data.delete_many(where=_url_filter(url)), name="Weaviate")

    def clear(self) -> None:
        """Delete every chunk in the tenant, for example before a new evaluation run."""
        everything = Filter.by_property("chunk_index").greater_or_equal(0)
        self._retrier.call(lambda: self._collection.data.delete_many(where=everything), name="Weaviate")

    def list_sources(self) -> dict[str, int]:
        """Count the stored chunks per source.

        Returns:
            A mapping from the address of each source to the number of chunks stored for it.
        """
        result = self._retrier.call(
            lambda: self._collection.aggregate.over_all(group_by=GroupByAggregate(prop="url")), name="Weaviate"
        )
        return {str(group.grouped_by.value): group.total_count or 0 for group in result.groups}

    def count_chunks(self) -> int:
        """Count all the chunks in the tenant, for example to stay under the object limit of a free plan."""
        result = self._retrier.call(lambda: self._collection.aggregate.over_all(total_count=True), name="Weaviate")
        return result.total_count or 0

    def has_source(self, url: str) -> bool:
        """Check whether any chunks are stored for a source.

        Args:
            url: The address of the source.

        Returns:
            ``True`` if at least one chunk exists.
        """
        response = self._retrier.call(
            lambda: self._collection.query.fetch_objects(filters=_url_filter(url), limit=1, return_properties=[]),
            name="Weaviate",
        )
        return bool(response.objects)

    def _record_values(self, record: SourceRecord) -> dict[str, Any]:
        """The values of a record that the collection has a field for."""
        values: dict[str, Any] = {}
        if self._keeps_loaded_at:
            values[AGE_PROPERTY] = record.loaded_at or datetime.now(UTC)
        if self._keeps_versions:
            marks = (record.etag, record.last_modified, record.content_hash)
            values.update({name: value for name, value in zip(VERSION_PROPERTIES, marks, strict=True) if value})
        return values

    def source_record(self, url: str) -> SourceRecord | None:
        """Find out what is kept about the stored copy of a source.

        Args:
            url: The address of the source.

        Returns:
            What is kept about the copy, or ``None`` when the source is not stored. A part that is not known is
            ``None`` in the record: the collection has no field for it, or the copy was stored before the field was
            added.
        """
        names = [
            *([AGE_PROPERTY] if self._keeps_loaded_at else []),
            *(VERSION_PROPERTIES if self._keeps_versions else []),
            *([LINKS_PROPERTY] if self._keeps_links else []),
        ]
        first_chunk = Filter.all_of([_url_filter(url), Filter.by_property("chunk_index").equal(FIRST_CHUNK)])
        response = self._retrier.call(
            lambda: self._collection.query.fetch_objects(filters=first_chunk, limit=1, return_properties=names),
            name="Weaviate",
        )
        if not response.objects:
            return None
        found = response.objects[0].properties
        return SourceRecord(
            loaded_at=_as_datetime(found.get(AGE_PROPERTY)),
            etag=_as_text(found.get(ETAG_PROPERTY)),
            last_modified=_as_text(found.get(LAST_MODIFIED_PROPERTY)),
            content_hash=_as_text(found.get(HASH_PROPERTY)),
            links=links_from_json(_as_text(found.get(LINKS_PROPERTY))),
        )

    def source_title(self, url: str) -> str | None:
        """Find the title of a stored source.

        Args:
            url: The address of the source.

        Returns:
            The title, or ``None`` when the source is not stored or has no title.
        """
        first_chunk = Filter.all_of([_url_filter(url), Filter.by_property("chunk_index").equal(FIRST_CHUNK)])
        response = self._retrier.call(
            lambda: self._collection.query.fetch_objects(filters=first_chunk, limit=1, return_properties=["title"]),
            name="Weaviate",
        )
        if not response.objects:
            return None
        return _as_text(response.objects[0].properties.get("title"))

    def fetch_section(self, url: str, heading: str, limit: int) -> list[RetrievedChunk]:
        """Read the chunks of one section of a stored source, in the order of the source.

        The section is found by its heading, which Weaviate matches by its words, so the heading of every chunk that
        comes back is compared again with the one asked for, and a section with a similar heading is left out.

        Args:
            url: The address of the source.
            heading: The heading of the section, for example ``Page 5`` for a page of a PDF file.
            limit: The most chunks to return.

        Returns:
            The chunks of the section, first chunk first. Empty when the source has no such section.
        """
        section = Filter.all_of([_url_filter(url), Filter.by_property("heading").equal(heading)])
        response = self._retrier.call(
            lambda: self._collection.query.fetch_objects(
                filters=section,
                limit=limit,
                sort=Sort.by_property("chunk_index"),
                return_properties=["url", "title", "heading", "text", "chunk_index"],
            ),
            name="Weaviate",
        )
        chunks = [_retrieved_chunk(found.properties, None) for found in response.objects]
        return [chunk for chunk in chunks if chunk.heading == heading]

    def fetch_chunks(self, url: str, indexes: Sequence[int]) -> list[RetrievedChunk]:
        """Read chunks of a stored source by their place in it, for example the ones next to a chunk.

        Args:
            url: The address of the source.
            indexes: The places of the chunks, which start at 0. A place that has no chunk, such as one after the end
                of the source, is left out of the result.

        Returns:
            The chunks that exist, first chunk first.
        """
        wanted = sorted(set(indexes))
        if not wanted:
            return []
        place = Filter.all_of([_url_filter(url), Filter.by_property("chunk_index").contains_any(wanted)])
        response = self._retrier.call(
            lambda: self._collection.query.fetch_objects(
                filters=place,
                limit=len(wanted),
                sort=Sort.by_property("chunk_index"),
                return_properties=["url", "title", "heading", "text", "chunk_index"],
            ),
            name="Weaviate",
        )
        chunks = [_retrieved_chunk(found.properties, None) for found in response.objects]
        return [chunk for chunk in chunks if chunk.url == url and chunk.chunk_index in wanted]

    def search(
        self,
        query: str,
        vector: Sequence[float],
        limit: int,
        alpha: float,
        urls: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search the stored chunks: BM25 on the text and heading blended with vector similarity.

        Args:
            query: The question, for the keyword part of the search.
            vector: The embedding of the question.
            limit: The most chunks to return.
            alpha: The blend of the two searches; 1 is the vector search only and 0 the keyword search only.
            urls: The sources to search in. ``None`` searches all of them, and an empty list finds nothing.

        Returns:
            The best chunks, best first.
        """
        if urls is not None and not urls:
            return []
        filters: Any = None
        if urls is not None:
            filters = _url_filter(urls[0]) if len(urls) == 1 else Filter.by_property("url").contains_any(list(urls))
        response = self._retrier.call(
            lambda: self._collection.query.hybrid(
                query=query,
                vector=list(vector),
                alpha=alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                query_properties=["text", "heading"],
                filters=filters,
                limit=limit,
                return_properties=["url", "title", "heading", "text", "chunk_index"],
                return_metadata=MetadataQuery(score=True),
            ),
            name="Weaviate",
        )
        return [_retrieved_chunk(found.properties, found.metadata.score) for found in response.objects]

    def close(self) -> None:
        """Close the connection to Weaviate, if this store owns it."""
        if self._client is not None:
            self._client.close()
