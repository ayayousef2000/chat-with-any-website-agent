"""The links of a page that the agent may follow, and the rules that decide which ones.

A page lists links to other pages of its site. The agent can follow some of them to find an answer that is not on the
page it was given. Which links it may follow is decided here, in plain code, before a model sees any of them: a link
must stay on the site of the page it was found on, use a standard web address, and lead to something that can be read.
A model that chooses among the links that remain only gets numbers to choose from, so it cannot be led to an address
that is not on this list. Link text comes from the page, so it is only ever data.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit

from chat_with_any_website_agent.fetching import ALLOWED_PORTS

if TYPE_CHECKING:
    from collections.abc import Iterable

# The most links kept from one page, and the longest text kept for a link. Menus and footers repeat the same links, and
# a page of a very large site can have thousands, so what is kept is bounded.
MAX_LINKS = 200
MAX_TEXT_CHARS = 120

# What the links lead to when it is not something to read: images, sound, video, archives, programs and the files that
# make a page look and work the way it does. A PDF file is kept, because it can be read.
_SKIPPED_EXTENSIONS = (
    # images
    *(".avif", ".bmp", ".gif", ".ico", ".jpeg", ".jpg", ".png", ".svg", ".webp"),
    # sound and video
    *(".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"),
    *(".avi", ".mkv", ".mov", ".mp4", ".webm", ".wmv"),
    # archives and programs
    *(".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip"),
    *(".apk", ".dmg", ".exe", ".iso", ".msi"),
    # parts of a page
    *(".css", ".js", ".json", ".map", ".woff", ".woff2", ".xml"),
    # office files, which are not read
    *(".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"),
)


@dataclass(frozen=True)
class Link:
    """A link of a page to another page of the same site.

    Attributes:
        url: The normalized address of the page it leads to, without a fragment.
        text: What the page says about it, with its white space made single and its length limited. It may be empty.
    """

    url: str
    text: str = ""


def site_of(url: str) -> str:
    """The site of an address: its host in lower case, without a leading ``www.``.

    Args:
        url: A web address.

    Returns:
        The site, or an empty text when the address has no host.
    """
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def collect_links(anchors: Iterable[tuple[str, str]], page_url: str) -> tuple[Link, ...]:
    """Choose the links of a page that may be followed, from all of its anchors.

    A link is kept when it is a web address on the same site as the page (``www.`` makes no difference), on a standard
    port, without a user name or password, and not to an image, a file of sound or video, an archive, a program, an
    office file or a part of the page itself. The address of the page is left out, and a fragment is dropped, so a
    link to another part of the same page is not a link to another page. Each address is kept once. The first
    ``MAX_LINKS`` are kept, in the order of the page.

    Args:
        anchors: The ``href`` and the shown text of every anchor of the page, as the page writes them.
        page_url: The address the page was loaded from, which relative links are read against.

    Returns:
        The links, in the order of the page.
    """
    here = _normalize(page_url)
    site = site_of(page_url)
    found: dict[str, str] = {}
    for href, shown in anchors:
        url = _target(href, page_url, site)
        if url is None or url == here:
            continue
        text = " ".join(shown.split())[:MAX_TEXT_CHARS].strip()
        if url not in found:
            if len(found) >= MAX_LINKS:
                break
            found[url] = text
        elif not found[url] and text:
            found[url] = text  # the first anchor of this address had no text, a later one has
    return tuple(Link(url=url, text=text) for url, text in found.items())


def links_to_json(links: Iterable[Link]) -> str:
    """Write links as a short JSON text, to keep them with a stored page."""
    return json.dumps([[link.url, link.text] for link in links], ensure_ascii=False, separators=(",", ":"))


def links_from_json(text: str | None) -> tuple[Link, ...]:
    """Read links that ``links_to_json`` wrote. Anything that is not such a text gives no links.

    Args:
        text: The JSON text, or ``None``.

    Returns:
        The links, without the entries that do not have an address and a text.
    """
    try:
        data = json.loads(text or "[]")
    except ValueError:
        return ()
    if not isinstance(data, list):
        return ()
    links = []
    for item in data:
        match item:
            case [str() as url, str() as shown] if url:
                links.append(Link(url=url, text=shown))
    return tuple(links)


def _target(href: str, page_url: str, site: str) -> str | None:
    """The normalized address a link leads to, or ``None`` when it may not be followed."""
    href = href.strip()
    if not href or href.startswith("#"):
        return None
    parts = urlsplit(urljoin(page_url, href))
    if parts.scheme not in {"http", "https"} or not parts.hostname or site_of(parts.geturl()) != site:
        return None
    if parts.username or parts.password:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is not None and port not in ALLOWED_PORTS:
        return None
    if parts.path.lower().endswith(_SKIPPED_EXTENSIONS):
        return None
    return _normalize(parts.geturl())


def _normalize(url: str) -> str:
    """An address without its fragment, with a lower-case host, and with ``/`` for an empty path."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))
