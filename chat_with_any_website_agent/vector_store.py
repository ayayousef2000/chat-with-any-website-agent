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
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateConnectionError, WeaviateTimeoutError

from chat_with_any_website_agent.errors import StoreError
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

# The field of every chunk that says when its source was loaded, so that an old copy can be loaded again.
AGE_PROPERTY = "loaded_at"

logger = logging.getLogger(__name__)


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
                Property(name=AGE_PROPERTY, data_type=DataType.DATE),
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


def ensure_age_property(collection: Any) -> bool:
    """Add the field that says when a source was loaded, if the collection does not have it yet.

    The change only adds a field. No chunk is changed or deleted, and no embedding is made. The chunks that were
    stored before have no value in the field, so their age is not known.

    Args:
        collection: The collection, not bound to a tenant.

    Returns:
        ``True`` if the field was added, ``False`` if the collection had it already.
    """
    if AGE_PROPERTY in property_names(collection):
        return False
    collection.config.add_property(Property(name=AGE_PROPERTY, data_type=DataType.DATE))
    return True


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
    ) -> None:
        """Create a store over a collection that is already bound to a tenant.

        Args:
            collection: The collection for the tenant, as ``open_tenant`` returns it.
            client: The connected client, closed by ``close``. Not needed when something else closes it.
            retrier: How to try again after a failed connection. The default suits a person who is waiting.
            keeps_loaded_at: Whether the collection has the field that says when a source was loaded. When it does
                not, nothing is written to that field and the age of a source is not known.
        """
        self._collection = collection
        self._client = client
        self._retrier = retrier or Retrier(retryable=WEAVIATE_RETRYABLE)
        self._keeps_loaded_at = keeps_loaded_at

    @property
    def keeps_loaded_at(self) -> bool:
        """Whether the collection keeps when each source was loaded."""
        return self._keeps_loaded_at

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
            keeps_loaded_at = AGE_PROPERTY in property_names(client.collections.use(collection_name))
        except Exception:
            client.close()  # a failed start must not leave the connection open
            raise
        if not keeps_loaded_at:
            logger.warning(
                "The collection %s has no field %s, so the age of the stored sources is not known. "
                "Run 'agent-smoke --prepare' once to add it.",
                collection_name,
                AGE_PROPERTY,
            )
        return cls(collection, client=client, retrier=retrier, keeps_loaded_at=keeps_loaded_at)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def replace_source(
        self, url: str, title: str, chunks: Sequence[StoredChunk], loaded_at: datetime | None = None
    ) -> None:
        """Remove any chunks previously stored for a source and store the new ones.

        Args:
            url: The address of the source.
            title: The title of the source.
            chunks: The chunks with their embeddings, in reading order.
            loaded_at: When the source was loaded, now when it is not given. It is only kept when the collection has
                the field for it.

        Raises:
            StoreError: If Weaviate refuses to store some of the chunks.
            ServiceBusyError: If Weaviate cannot be reached after the retries.
        """
        age: dict[str, Any] = {AGE_PROPERTY: loaded_at or datetime.now(UTC)} if self._keeps_loaded_at else {}
        objects = [
            DataObject(
                properties={
                    "url": url,
                    "title": title,
                    "heading": chunk.heading,
                    "chunk_index": index,
                    "text": chunk.text,
                    **age,
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

    def loaded_at(self, url: str) -> datetime | None:
        """Find out when a source was loaded.

        Args:
            url: The address of the source.

        Returns:
            The time the stored copy was made, or ``None`` when it is not known: the collection has no field for it,
            the source is not stored, or its chunks were stored before the field existed.
        """
        if not self._keeps_loaded_at:
            return None
        response = self._retrier.call(
            lambda: self._collection.query.fetch_objects(
                filters=_url_filter(url), limit=1, return_properties=[AGE_PROPERTY]
            ),
            name="Weaviate",
        )
        if not response.objects:
            return None
        return _as_datetime(response.objects[0].properties.get(AGE_PROPERTY))

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
        return [
            RetrievedChunk(
                url=str(found.properties["url"]),
                title=str(found.properties["title"]),
                heading=str(found.properties.get("heading") or ""),
                text=str(found.properties["text"]),
                chunk_index=int(found.properties["chunk_index"]),
                score=found.metadata.score,
            )
            for found in response.objects
        ]

    def close(self) -> None:
        """Close the connection to Weaviate, if this store owns it."""
        if self._client is not None:
            self._client.close()
