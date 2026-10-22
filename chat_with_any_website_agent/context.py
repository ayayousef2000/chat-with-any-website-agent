"""Widening the excerpts: the text next to the best ones.

A chunk ends where its size limit does, so a list, a table or a paragraph can be cut between two chunks, and the excerpt
that was found holds only one half of it. The other half is in the chunk next to it. When the excerpts are not enough,
the chunks before and after the best ones are added, which is cheaper and more precise than a new search: the text is
known to be about the same thing, and it is the text that the source has at that place. Each added excerpt keeps its own
heading, so a citation still names the section, or the page of a PDF file, that the text is from.
"""

from typing import TYPE_CHECKING, Protocol

from chat_with_any_website_agent.retriever import document_from_chunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document

    from chat_with_any_website_agent.vector_store import RetrievedChunk

# How many of the best excerpts are widened, and the most excerpts that are added at once, which keeps the prompt of the
# model within the limits of a free plan.
DEFAULT_BEST = 2
DEFAULT_MAX_ADDED = 4


class ChunkStore(Protocol):
    """What the expander needs from the store; ``WeaviateStore`` provides it."""

    def fetch_chunks(self, url: str, indexes: Sequence[int]) -> list[RetrievedChunk]:
        """Read chunks of a stored source by their place in it."""
        ...


class ContextExpander:
    """Finds the chunks next to the best excerpts."""

    def __init__(self, store: ChunkStore, *, best: int = DEFAULT_BEST, max_added: int = DEFAULT_MAX_ADDED) -> None:
        """Create an expander.

        Args:
            store: Where the chunks are stored, for example a ``WeaviateStore``.
            best: How many of the first excerpts are widened. The excerpts are given best first.
            max_added: The most excerpts to add.
        """
        self._store = store
        self._best = best
        self._max_added = max_added

    def around(self, documents: Sequence[Document], reach: int = 1) -> list[Document]:
        """The chunks within ``reach`` places of the best excerpts that are not among the excerpts yet.

        Args:
            documents: The excerpts, best first. Their metadata has the ``url`` and ``chunk_index``.
            reach: How many places away from an excerpt to look: 1 is the chunk before and the chunk after it.

        Returns:
            The chunks to add, nearest to the best excerpts first and the one before an excerpt before the one after
            it, at most ``max_added``. Empty when the best excerpts have no neighbours that are not excerpts already,
            for example because they are the whole source.

        Raises:
            ServiceBusyError: If Weaviate keeps failing after the retries.
        """
        have = {(str(document.metadata["url"]), int(document.metadata["chunk_index"])) for document in documents}
        best = [(str(d.metadata["url"]), int(d.metadata["chunk_index"])) for d in documents[: self._best]]
        order: list[tuple[str, int]] = []
        for distance in range(1, reach + 1):
            for url, index in best:
                for place in (index - distance, index + distance):
                    if place >= 0 and (url, place) not in have and (url, place) not in order:
                        order.append((url, place))
        if not order:
            return []
        found: dict[tuple[str, int], RetrievedChunk] = {}
        for url in dict.fromkeys(url for url, _ in order):
            for chunk in self._store.fetch_chunks(url, [place for source, place in order if source == url]):
                found[(chunk.url, chunk.chunk_index)] = chunk
        return [document_from_chunk(found[place], None) for place in order if place in found][: self._max_added]
