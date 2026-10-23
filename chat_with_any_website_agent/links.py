"""The links of a page that the agent may follow, and the rules that decide which ones.

A page lists links to other pages of its site. The agent can follow some of them to find an answer that is not on the
page it was given. Which links it may follow is decided here, in plain code, before a model sees any of them: a link
must stay on the site of the page it was found on, use a standard web address, and lead to something that can be read.
A model that chooses among the links that remain only gets numbers to choose from, so it cannot be led to an address
that is not on this list. Link text comes from the page, so it is only ever data.
"""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit

from chat_with_any_website_agent.fetching import ALLOWED_PORTS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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


# The words that every question has and that say nothing about which page holds the answer.
_COMMON_WORDS = frozenset(
    {
        *("about", "after", "also", "and", "any", "are", "been", "can", "could", "did", "does", "for", "from"),
        *("has", "have", "how", "into", "its", "not", "that", "the", "their", "there", "these", "this", "was"),
        *("were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your"),
    }
)
_STEM_CHARS = 6
# The two-letter codes of languages (ISO 639-1). A translated site puts one in the address of each page ("/book/de/v2").
_LANGUAGE_CODES = frozenset(
    re.findall(
        r"[a-z]{2}",
        "aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo "
        "br bs ca ce ch co cr cs cu cv cy da de dv dz ee el en eo es "
        "et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr "
        "ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj "
        "kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv "
        "mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv "
        "ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd "
        "se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti "
        "tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi "
        "yo za zh zu",
    )
)
# A language with a script or a country after it: "zh-hans", "pt-br", "pt_BR".
_LANGUAGE_TAG = re.compile(r"[a-z]{2}(?:[-_][a-z0-9]{2,4})?")


def is_other_language(page_url: str, link_url: str) -> bool:
    """Whether a link leads to a page in another language than the page it was found on.

    A site that is translated lists the other languages of a page among its links, and a model that is asked what is
    missing may choose one of them: it holds the same text in a language that the question is not asked in, and its
    excerpts cannot be checked against an answer in the language of the question. Two addresses are the same place in
    two languages when they are the same up to a segment where both have a language code, as "/book/en/v2/hooks" and
    "/book/de/v2/haken" are. Only real language codes count, so "/docs/js/intro" and "/docs/py/intro" are not.

    Args:
        page_url: The address of the page the link was found on.
        link_url: The address the link leads to.

    Returns:
        ``True`` when the first segment that the addresses do not share is a language code in both.
    """
    for here, there in zip(_segments(page_url), _segments(link_url), strict=False):
        if here != there:
            return _is_language(here) and _is_language(there)
    return False


def _segments(url: str) -> list[str]:
    return urlsplit(url).path.lower().strip("/").split("/")


def _is_language(segment: str) -> bool:
    return _LANGUAGE_TAG.fullmatch(segment) is not None and segment[:2] in _LANGUAGE_CODES


def shortlist_links(links: Sequence[Link], topic: str, limit: int, near: Sequence[str] = ()) -> tuple[Link, ...]:
    """Choose the links worth showing to a model: the ones that share the most words with a topic.

    A page can have hundreds of links, and every one shown to the model costs tokens. When there are more links than
    ``limit``, the ones whose text or address share the most words with the topic come first (word endings are
    ignored, so ``statements`` matches ``statement``). Among links that match equally, the ones near the pages in
    ``near`` come first, and then the order of the page decides. The links near those pages are also listed before the
    others, because the answer to a question about a part of a website is usually in the same part, and a model gives
    the first links of a list more weight. When all the links fit and ``near`` is empty, they are returned as they are,
    in the order of the page.

    Args:
        links: The candidate links, in the order of the page.
        topic: What is being looked for, for example the question and what the pages read so far were missing.
        limit: The most links to return.
        near: The addresses of the pages the question was asked about. The part of a page is its directory, the
            address up to the last slash: ``/3/tutorial/index.html`` is in ``/3/tutorial/``, so a link to
            ``/3/tutorial/errors.html`` is near it and a link to ``/3/reference/index.html`` is not. A page at the top
            of the site (``/`` or ``/index.html``) says nothing about a part of it, so no link is near it.

    Returns:
        At most ``limit`` links.
    """
    if limit <= 0:
        return ()
    sections = _sections(near)
    chosen = list(links)
    if len(chosen) > limit:
        wanted = _stems(topic)
        scored = [
            (-len(wanted & _stems(f"{link.text} {urlsplit(link.url).path}")), not _is_in(link, sections), index)
            for index, link in enumerate(chosen)
        ]
        chosen = [chosen[index] for *_, index in sorted(scored)[:limit]]
    if sections:  # a stable sort: each group keeps its order
        chosen.sort(key=lambda link: not _is_in(link, sections))
    return tuple(chosen)


def _sections(pages: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """The site and directory of each page, leaving out the pages at the top of their site."""
    sections = []
    for page in pages:
        directory = urlsplit(page).path.rpartition("/")[0] + "/"
        if directory != "/":
            sections.append((site_of(page), directory))
    return tuple(sections)


def _is_in(link: Link, sections: Iterable[tuple[str, str]]) -> bool:
    return any(
        site_of(link.url) == site and urlsplit(link.url).path.startswith(directory) for site, directory in sections
    )


def _stems(text: str) -> set[str]:
    """The words of a text without their endings, leaving out the ones that every question has."""
    words = re.findall(r"\w+", text.lower().replace("_", " "))
    return {
        word[:_STEM_CHARS] if len(word) > _STEM_CHARS else word.removesuffix("s")
        for word in words
        if len(word) > 2 and word not in _COMMON_WORDS
    }
