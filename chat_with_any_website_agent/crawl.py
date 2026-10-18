"""Loading the pages of a site before any question is asked: the baseline that the link-following agent is compared to.

A crawl starts at one page and follows the links of every page it loads, level by level, in the order the pages list
them. It uses the same rules as the agent: the same site only, the site's ``robots.txt``, no redirect to another
site. It stops at a number of pages and at a number of links away from the first page, because every page costs
embedding calls.
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat_with_any_website_agent.graph import LinkSource


@dataclass(frozen=True)
class CrawlLimits:
    """How far a crawl goes.

    Attributes:
        max_pages: The most pages loaded, not counting the first one.
        max_depth: How many links away from the first page a page may be. Zero loads nothing but the first page.
    """

    max_pages: int
    max_depth: int


def crawl_site(links: LinkSource, start: str, limits: CrawlLimits) -> list[str]:
    """Load the pages of a site, breadth first, starting from a page that is already stored.

    Args:
        links: Reads the links of the stored pages and opens them, for example a ``LinkFollower``.
        start: The address the first page is stored under.
        limits: How many pages and how many levels to load.

    Returns:
        The addresses of the pages that were loaded, in the order they were loaded, without the first page.

    Raises:
        ServiceBusyError: If Cohere or Weaviate keeps failing after the retries.
        BudgetExceededError: If the run's limit of Cohere calls is reached.
    """
    seen = {start}
    loaded: list[str] = []
    queue = deque([(start, 0)])
    while queue and len(loaded) < limits.max_pages:
        page, depth = queue.popleft()
        if depth >= limits.max_depth:
            continue
        for link in links.links_of(page):
            if link.url in seen:
                continue
            seen.add(link.url)
            if len(loaded) >= limits.max_pages:
                break
            result = links.follow(link.url)
            if result.loaded and result.url not in loaded and result.url != start:
                loaded.append(result.url)
                queue.append((result.url, depth + 1))
    return loaded
