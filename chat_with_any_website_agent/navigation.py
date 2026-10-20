"""Following a link of a stored page: the checks, the loading and the links of the page that was loaded.

The agent reads the links of the pages it has stored and decides which one to open next. This module opens it, and
holds the rules that do not depend on the model: the site's ``robots.txt`` must allow the page, the page must stay on
the site of the link even after a redirect, and the page is stored like any other source, so that its text can be
searched and its own links can be followed in turn. A page that cannot be opened is a result, not an error: the agent
notes why and goes on with the others.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from chat_with_any_website_agent.ingest import SOURCE_ERRORS
from chat_with_any_website_agent.links import Link, site_of

if TYPE_CHECKING:
    from chat_with_any_website_agent.ingest import IngestResult
    from chat_with_any_website_agent.robots import RobotsChecker
    from chat_with_any_website_agent.vector_store import SourceRecord

logger = logging.getLogger(__name__)

ROBOTS_REFUSAL = "the site's robots.txt does not allow programs to open it"


class PageIngestor(Protocol):
    """What following a link needs from the ingestor; ``SourceIngestor`` provides it."""

    def ingest(self, url: str, *, check: bool = True, site: str | None = None) -> IngestResult:
        """Make a source ready for questions."""
        ...


class LinkStore(Protocol):
    """What following a link needs from the store; ``WeaviateStore`` provides it."""

    def source_record(self, url: str) -> SourceRecord | None:
        """What is kept about the stored copy of a source."""
        ...


@dataclass(frozen=True)
class FollowResult:
    """What came of opening a link.

    Attributes:
        url: The address of the page: the one it is stored under when it was loaded, otherwise the one asked for.
        loaded: ``True`` when the page is stored and can be searched.
        title: The title of the page, when it was newly stored.
        chunk_count: How many chunks were stored, when the page was newly stored.
        note: Why the page was not loaded, in plain words. Empty when it was.
    """

    url: str
    loaded: bool
    title: str | None = None
    chunk_count: int | None = None
    note: str = ""


class LinkFollower:
    """Opens the links of stored pages, within the rules of the site."""

    def __init__(self, ingestor: PageIngestor, store: LinkStore, robots: RobotsChecker, *, check: bool = True) -> None:
        """Create a follower.

        Args:
            ingestor: Loads a page, for example a ``SourceIngestor``.
            store: Keeps the links of the pages, for example a ``WeaviateStore``.
            robots: Tells which pages the sites allow programs to open.
            check: Whether to ask the website if a page that is already stored changed. ``False`` uses a stored copy as
                it is, which makes the answers of one run independent of when each page was asked about.
        """
        self._ingestor = ingestor
        self._store = store
        self._robots = robots
        self._check = check

    def links_of(self, url: str) -> tuple[Link, ...]:
        """The links of a stored page, in the order of the page.

        Args:
            url: The address the page is stored under.

        Returns:
            The links, or none when the page is not stored or was stored without them.
        """
        record = self._store.source_record(url)
        return record.links if record is not None else ()

    def follow(self, url: str) -> FollowResult:
        """Open a link: check the site's rules, then load the page unless a copy of it is already stored.

        Args:
            url: The address of the page, from the links of a stored page.

        Returns:
            Whether the page is stored now, and if not, why.

        Raises:
            ServiceBusyError: If Cohere or Weaviate keeps failing after the retries.
            BudgetExceededError: If the run's limit of Cohere calls is reached.
        """
        if not self._robots.allows(url):
            return FollowResult(url=url, loaded=False, note=ROBOTS_REFUSAL)
        self._robots.wait_turn(url)
        try:
            result = self._ingestor.ingest(url, check=self._check, site=site_of(url))
        except SOURCE_ERRORS as error:
            logger.info("A link could not be followed: %s", type(error).__name__)
            return FollowResult(url=url, loaded=False, note=str(error))
        return FollowResult(url=result.url, loaded=True, title=result.title, chunk_count=result.chunk_count)
