"""The rules a website gives to programs that read it: its ``robots.txt``.

The agent opens pages that the user did not name, by following links. A site says in ``robots.txt`` which of its pages
programs may open, and the agent follows that for every link it follows. The address the user gives is the user's own
request and is not checked here.

The file is read as RFC 9309 describes, including ``*`` and ``$`` in rules, and it is asked for once for each site
for as long as the checker lives. What its answer means:

* A file that was read gives its rules.
* A file that is not there, or that the site does not allow to be read (any 4xx answer, except 429), means that
  nothing is forbidden.
* A site that cannot be asked, that answers with an error of its own (5xx), that asks to slow down (429) or that
  redirects without end means that nothing is allowed for now. Doubt goes to the site, not to the agent.

The file is fetched with the same checks as every page: public addresses only, the connection goes to the address
that was checked, and the size and the time are limited.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from protego import Protego

from chat_with_any_website_agent.errors import FetchError, InvalidURLError
from chat_with_any_website_agent.fetching import DEFAULT_TIMEOUT, FetchedText, fetch_text

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# The name that sites use in their rules to address this program: the product part of its User-Agent.
USER_AGENT_PRODUCT = "ChatWithAnyWebsiteAgent"

# RFC 9309 asks programs to read at least 500 KiB of the file and allows them to ignore the rest.
MAX_ROBOTS_BYTES = 500 * 1024

# The longest wait between two requests to one site. A site that asks for more is still obeyed, with this wait, so
# that a question is not held up for minutes by a value in a file.
MAX_CRAWL_DELAY = 10.0

_SUCCESS = range(200, 300)
_UNAVAILABLE = range(400, 500)  # the file is not there, or the site does not give it to programs
_TOO_MANY_REQUESTS = 429  # a 4xx answer that asks to wait, not one that says there is no file


@dataclass(frozen=True)
class _Verdict:
    """What is known about the rules of one site."""

    rules: Protego | None  # the rules that were read; ``None`` when there is no file to follow
    allowed_without_rules: bool  # the answer for every page when there are no rules


class RobotsChecker:
    """Says whether the agent may open a page, by the ``robots.txt`` of the site of that page.

    The file of a site is asked for the first time a page of the site is checked and is kept after that, so a question
    that follows several links of one site asks for the file once. The checker is meant to live for one question or
    one run, so a changed file is read again by the next one.
    """

    def __init__(
        self,
        fetch: Callable[[str], FetchedText] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a checker.

        Args:
            fetch: How to download the file, given its address. It defaults to the safe download of this project;
                tests give a fake.
            timeout: The time limit of one step of the download, in seconds, for the default download.
            sleep: How to wait, for the pause between requests to one site.
            clock: A clock that only moves forward, in seconds, for that pause.
        """
        self._fetch = fetch or (lambda url: fetch_text(url, timeout=timeout, max_bytes=MAX_ROBOTS_BYTES))
        self._sleep = sleep
        self._clock = clock
        self._verdicts: dict[str, _Verdict] = {}
        self._last_request: dict[str, float] = {}

    def allows(self, url: str) -> bool:
        """Tell whether the rules of the site of ``url`` allow the agent to open it.

        Args:
            url: The address of a page, as it would be opened.

        Returns:
            ``True`` if the page may be opened. An address that is not a web address is not allowed.
        """
        origin = _origin_of(url)
        if origin is None:
            return False
        verdict = self._verdict(origin)
        if verdict.rules is None:
            return verdict.allowed_without_rules
        return bool(verdict.rules.can_fetch(url, USER_AGENT_PRODUCT))

    def crawl_delay(self, url: str) -> float:
        """The pause that the site asks for between two requests, in seconds.

        Args:
            url: The address of a page of the site.

        Returns:
            The pause, at most ``MAX_CRAWL_DELAY``; zero when the site asks for none or its rules are not known.
        """
        origin = _origin_of(url)
        if origin is None:
            return 0.0
        rules = self._verdict(origin).rules
        delay = rules.crawl_delay(USER_AGENT_PRODUCT) if rules is not None else None
        return min(max(float(delay), 0.0), MAX_CRAWL_DELAY) if delay else 0.0

    def wait_turn(self, url: str) -> None:
        """Wait, if the site asked for a pause, until the pause since the last request to the site has passed.

        Call it just before a request to a page. It also notes the time of this request.

        Args:
            url: The address of the page that is about to be requested.
        """
        origin = _origin_of(url)
        if origin is None:
            return
        pause = self.crawl_delay(url)
        last = self._last_request.get(origin)
        if pause and last is not None:
            remaining = pause - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request[origin] = self._clock()

    def _verdict(self, origin: str) -> _Verdict:
        if origin not in self._verdicts:
            self._verdicts[origin] = self._read(origin)
        return self._verdicts[origin]

    def _read(self, origin: str) -> _Verdict:
        """Ask a site for its file once and decide what the answer means."""
        parts = urlsplit(origin)
        host = parts.hostname or "an unknown host"
        try:
            answer = self._fetch(urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", "")))
        except (FetchError, InvalidURLError) as error:
            logger.warning("The robots.txt of %s could not be read (%s), so its pages are not followed", host, error)
            return _Verdict(rules=None, allowed_without_rules=False)
        if answer.status in _SUCCESS:
            return _Verdict(rules=Protego.parse(answer.body), allowed_without_rules=True)
        if answer.status in _UNAVAILABLE and answer.status != _TOO_MANY_REQUESTS:
            logger.info("%s has no robots.txt to follow (status %s)", host, answer.status)
            return _Verdict(rules=None, allowed_without_rules=True)
        logger.warning(
            "The robots.txt of %s answered with status %s, so its pages are not followed", host, answer.status
        )
        return _Verdict(rules=None, allowed_without_rules=False)


def _origin_of(url: str) -> str | None:
    """The scheme and the host (with its port) that a file of rules belongs to, or ``None`` if not a web address."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    return f"{parts.scheme}://{parts.netloc.lower()}"
