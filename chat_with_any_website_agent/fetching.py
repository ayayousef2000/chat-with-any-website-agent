"""Safe downloading of web pages and PDF files.

The agent follows links it finds on pages, so every address it fetches is treated as untrusted. An address is
looked up once, every address it resolves to must be public, and the connection goes to the address that was checked,
not to a second lookup of the name, so a name server cannot lead the connection to a private address (DNS
rebinding). Each redirect is checked again in the same way. Only the standard web ports are used, an address with a
user name or password is refused, and the size and the total time of a download are limited whatever the server says.

A page that was downloaded before can be asked for again only if it changed, with the standard conditional request
(``If-None-Match`` and ``If-Modified-Since``). The same checks apply to that request.

A plain text file, such as the ``robots.txt`` of a site, is downloaded with the same checks, and its status is
returned instead of being turned into an error.
"""

import ipaddress
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from chat_with_any_website_agent import __version__
from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

PROJECT_URL = "https://github.com/ayayousef2000/chat-with-any-website-agent"
USER_AGENT = f"ChatWithAnyWebsiteAgent/{__version__} (+{PROJECT_URL})"

# Messages that are safe to show to a person or to hand to the model as a tool result: what happened, and what to do.
INVALID_URL_MESSAGE = "That is not a valid web address. Use a link that starts with https://."
PRIVATE_ADDRESS_MESSAGE = (
    "That address is on a private network and cannot be loaded. Use the address of a public web page."
)
CREDENTIALS_MESSAGE = "Addresses that contain a user name or password cannot be loaded."
DISALLOWED_PORT_MESSAGE = "Only the standard web ports (80 and 443) can be used."
TIMEOUT_MESSAGE = "The website took too long to respond. Try again in a moment."
UNREACHABLE_MESSAGE = "The website could not be reached. Check the address and try again."
TOO_LARGE_MESSAGE = "This page is too large to load."
TOO_MANY_REDIRECTS_MESSAGE = "That address redirects too many times. Try the final address of the page."
PDF_DISABLED_MESSAGE = "That link is a PDF file, and reading PDF files is switched off."

DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 5_000_000

# Only the standard web ports are used: a link on a page must not make the agent knock on other services.
ALLOWED_PORTS = frozenset({80, 443})

# Limits of one download: redirects followed, addresses tried for one name, and the whole time as a multiple of the
# timeout of a single step (which a site sending a few bytes at a time would otherwise stretch without end).
MAX_REDIRECTS = 5
MAX_ADDRESSES_TRIED = 4
TOTAL_TIME_FACTOR = 2

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_NOT_MODIFIED = 304
# A version mark (an ETag or a date) is short. A longer one is not used, because it comes from a website.
MAX_VALIDATOR_CHARS = 256
_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_FILE_KINDS = {"image": "an image", "video": "a video", "audio": "an audio file"}
# Addresses in this range stand for IPv4 addresses (RFC 6052); 64:ff9b::a00:1 reaches 10.0.0.1 through a NAT64 gateway.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _header_value(value: str | None) -> str | None:
    """A version mark that is safe to keep and to send back: short, plain printable text; otherwise ``None``."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_VALIDATOR_CHARS or not cleaned.isascii() or not cleaned.isprintable():
        return None
    return cleaned


@dataclass(frozen=True)
class Validators:
    """What a website said about the version of a page it sent, so that it can be asked later whether it changed.

    Attributes:
        etag: The ``ETag`` of the page, as the website wrote it.
        last_modified: The ``Last-Modified`` date of the page, as the website wrote it.
    """

    etag: str | None = None
    last_modified: str | None = None

    def request_headers(self) -> dict[str, str]:
        """The headers that ask a website to send a page only if it changed.

        A value that is empty, long or not plain printable text is left out. The values come from a website, and from
        storage when they were saved, so they are checked before they are sent back.

        Returns:
            The headers, which are none when nothing is known about the version.
        """
        headers: dict[str, str] = {}
        for name, value in (("If-None-Match", self.etag), ("If-Modified-Since", self.last_modified)):
            cleaned = _header_value(value)
            if cleaned:
                headers[name] = cleaned
        return headers


@dataclass(frozen=True)
class FetchedDocument:
    """A downloaded page or PDF file.

    Attributes:
        kind: ``"html"`` for a web page, ``"pdf"`` for a PDF file.
        body: The bytes of the response, within the size limit.
        url: The address the content came from, after redirects and without a fragment.
        validators: What the website said about the version of the document, to ask later whether it changed.
    """

    kind: Literal["html", "pdf"]
    body: bytes
    url: str
    validators: Validators = field(default_factory=Validators)


@dataclass(frozen=True)
class FetchedText:
    """A downloaded text file, with the status the website answered.

    Attributes:
        status: The HTTP status of the final response, after redirects. It is not turned into an error.
        body: The text, cut at the size limit and read as UTF-8 with unreadable bytes replaced.
        url: The address the text came from, after redirects and without a fragment.
    """

    status: int
    body: str
    url: str


def normalize_url(url: str) -> str:
    """Validate an address and drop its fragment so it can be used as a stable key.

    Args:
        url: The address. A missing scheme is treated as https.

    Returns:
        The address with a lowercase host, a path of at least ``/`` and no fragment.

    Raises:
        InvalidURLError: If the address is not a valid http(s) address, contains a user name or password, or uses a
            port other than 80 and 443.
    """
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    port: int | None = None
    try:
        parts = urlsplit(url)
        port = parts.port  # reading the port checks it; a non-numeric port raises ValueError
        valid = parts.scheme in {"http", "https"} and bool(parts.hostname)
    except ValueError:
        valid = False
    if not valid:
        raise InvalidURLError(INVALID_URL_MESSAGE)
    if parts.username is not None or parts.password is not None:
        raise InvalidURLError(CREDENTIALS_MESSAGE)
    if port is not None and port not in ALLOWED_PORTS:
        raise InvalidURLError(DISALLOWED_PORT_MESSAGE)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def is_public_address(address: str) -> bool:
    """Tell whether an address belongs to the public internet, also when it wraps an IPv4 address.

    An IPv6 address can carry an IPv4 one (IPv4-mapped, 6to4 and the NAT64 prefix); the wrapped address decides.

    Args:
        address: An IPv4 or IPv6 address as text.

    Returns:
        ``True`` if the address is globally routable.
    """
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in _NAT64_PREFIX:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None and not embedded.is_global:
            return False
    return ip.is_global


def resolve_public_addresses(host: str) -> list[str]:
    """Look a host up once and return its addresses, refusing a host that has any non-public address.

    Args:
        host: A host name or an IP address.

    Returns:
        The addresses of the host, without repeats.

    Raises:
        InvalidURLError: If the host cannot be found or has a private, loopback or otherwise non-public address.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidURLError(f"No website was found at {host}. Check the spelling and try again.") from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses or not all(is_public_address(address) for address in addresses):
        raise InvalidURLError(PRIVATE_ADDRESS_MESSAGE)
    return addresses


def fetch_document(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    pdf_max_bytes: int = 0,
) -> FetchedDocument:
    """Download a web page or, when ``pdf_max_bytes`` is above zero, a PDF file.

    Every address, including each redirect, is looked up once, checked and then connected to as checked. The size
    limits and a total time limit apply to the bytes that arrive, whatever the server says.

    Args:
        url: The address to download.
        timeout: The time limit of one step in seconds. The whole download may take twice as long.
        max_bytes: The largest web page in bytes.
        pdf_max_bytes: The largest PDF file in bytes; ``0`` refuses PDF files.

    Returns:
        The downloaded document.

    Raises:
        InvalidURLError: If an address is invalid, uses a disallowed port, has credentials or is not public.
        FetchError: If the download fails, takes too long, is too large or redirects too often.
        ExtractionError: If the address leads to something other than a web page or an allowed PDF file.
    """
    document = _download(url, Validators(), timeout, max_bytes, pdf_max_bytes)
    if document is None:  # pragma: no cover - only a conditional request can be answered with "not modified"
        raise AssertionError("a request without version marks cannot be answered with 'not modified'")
    return document


def fetch_if_changed(
    url: str,
    validators: Validators,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    pdf_max_bytes: int = 0,
) -> FetchedDocument | None:
    """Download a page again only if the website says that it changed since the version described by ``validators``.

    The request is made with the same checks as ``fetch_document``. When nothing is known about the version, it is
    an ordinary download.

    Args:
        url: The address to download.
        validators: What the website said about the version that is stored, from an earlier download.
        timeout: The time limit of one step in seconds. The whole download may take twice as long.
        max_bytes: The largest web page in bytes.
        pdf_max_bytes: The largest PDF file in bytes; ``0`` refuses PDF files.

    Returns:
        The downloaded document, or ``None`` when the website answered that the stored version is still the current
        one, in which case nothing was downloaded.

    Raises:
        InvalidURLError: If an address is invalid, uses a disallowed port, has credentials or is not public.
        FetchError: If the download fails, takes too long, is too large or redirects too often.
        ExtractionError: If the address leads to something other than a web page or an allowed PDF file.
    """
    return _download(url, validators, timeout, max_bytes, pdf_max_bytes)


def fetch_text(url: str, *, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = DEFAULT_MAX_BYTES) -> FetchedText:
    """Download a plain text file, such as ``robots.txt``, and say which status the website answered.

    The request has the same checks as ``fetch_document``: every address, including each redirect, is looked up once,
    must be public and is connected to as checked. Unlike a page, any status is a result, because what a missing or
    refused file means is for the caller to decide. A file larger than the limit is cut, not refused.

    Args:
        url: The address to download.
        timeout: The time limit of one step in seconds. The whole download may take twice as long.
        max_bytes: The most bytes to read.

    Returns:
        The final status and the text.

    Raises:
        InvalidURLError: If an address is invalid, uses a disallowed port, has credentials or is not public.
        FetchError: If the website cannot be reached, takes too long or redirects too often.
    """

    def read(response: httpx.Response, target: httpx.URL, deadline: float) -> FetchedText:
        body = bytearray()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) >= max_bytes:
                break
            _check_deadline(deadline)
        text = bytes(body[:max_bytes]).decode("utf-8", errors="replace")
        return FetchedText(status=response.status_code, body=text, url=str(target))

    return _get(url, {"Accept": "text/plain,*/*;q=0.1"}, timeout, read)


def _download(
    url: str, validators: Validators, timeout: float, max_bytes: int, pdf_max_bytes: int
) -> FetchedDocument | None:
    """Download a page or PDF file, or return ``None`` when a conditional request is answered with 'not modified'."""
    accept = "text/html,application/xhtml+xml" + (",application/pdf;q=0.9" if pdf_max_bytes > 0 else "")
    conditional = validators.request_headers()

    def read(response: httpx.Response, target: httpx.URL, deadline: float) -> FetchedDocument | None:
        if conditional and response.status_code == _NOT_MODIFIED:
            return None
        kind, body = _read_body(response, max_bytes, pdf_max_bytes, deadline)
        return FetchedDocument(kind=kind, body=body, url=str(target), validators=_validators_of(response))

    return _get(url, {"Accept": accept, **conditional}, timeout, read)


def _get[T](
    url: str, headers: dict[str, str], timeout: float, read: Callable[[httpx.Response, httpx.URL, float], T]
) -> T:
    """Send a GET request that is checked at every step, follow its redirects, and ``read`` the final response.

    ``read`` is called once, with the response that is not a redirect, the address it came from and the deadline of
    the whole request.
    """
    deadline = time.monotonic() + timeout * TOTAL_TIME_FACTOR
    current = url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            _check_deadline(deadline)
            target = httpx.URL(current).copy_with(fragment=None)
            _check_target(target)
            addresses = resolve_public_addresses(target.raw_host.decode("ascii"))
            # One client per address: a pooled connection is never reused for another name; the environment's proxy
            # settings are ignored because a proxy would do the connecting.
            with httpx.Client(
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, **headers},
                trust_env=False,
            ) as client:
                response = _send_pinned(client, target, addresses, deadline)
                try:
                    redirect = _redirect_target(response, target)
                    if redirect is None:
                        return read(response, target, deadline)
                finally:
                    response.close()
            current = str(redirect)
        raise FetchError(TOO_MANY_REDIRECTS_MESSAGE)
    except httpx.HTTPStatusError as exc:
        logger.info("%s answered with HTTP %s", _host_of(url), exc.response.status_code)
        raise FetchError(_status_message(exc.response.status_code)) from exc
    except httpx.TimeoutException as exc:
        raise FetchError(TIMEOUT_MESSAGE) from exc
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # The library's own wording is for developers; it stays out of the message shown to the user.
        logger.warning("Could not fetch a page from %s: %s", _host_of(url), type(exc).__name__)
        raise FetchError(UNREACHABLE_MESSAGE) from exc


def _validators_of(response: httpx.Response) -> Validators:
    """What the website said about the version of the page it sent."""
    return Validators(
        etag=_header_value(response.headers.get("etag")),
        last_modified=_header_value(response.headers.get("last-modified")),
    )


def _check_target(target: httpx.URL) -> None:
    """Refuse an address that is not http(s) on a standard port, or that carries a user name or password."""
    if target.scheme not in {"http", "https"}:
        raise FetchError(UNREACHABLE_MESSAGE)
    if target.userinfo:
        raise InvalidURLError(CREDENTIALS_MESSAGE)
    if target.port is not None and target.port not in ALLOWED_PORTS:
        raise InvalidURLError(DISALLOWED_PORT_MESSAGE)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise FetchError(TIMEOUT_MESSAGE)


def _send_pinned(client: httpx.Client, target: httpx.URL, addresses: list[str], deadline: float) -> httpx.Response:
    """Connect to the address that was checked, not to a fresh lookup of the name.

    The name is only used for the Host header and for the TLS check, so a second answer of the name server
    (DNS rebinding) cannot lead the connection to a private address.
    """
    host = target.raw_host.decode("ascii")
    failure: httpx.TransportError | None = None
    for address in addresses[:MAX_ADDRESSES_TRIED]:
        _check_deadline(deadline)
        request = client.build_request(
            "GET",
            target.copy_with(host=address),
            headers={"Host": target.netloc.decode("ascii")},
            extensions={"sni_hostname": host},
        )
        try:
            return client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            failure = exc
    if failure is None:  # pragma: no cover - the caller passes at least one address
        raise AssertionError("no address to connect to")
    raise failure


def _redirect_target(response: httpx.Response, current: httpx.URL) -> httpx.URL | None:
    """The address a redirect points to, or ``None`` when the response is not a redirect."""
    location = response.headers.get("location")
    if response.status_code not in _REDIRECT_STATUSES or not location:
        return None
    return current.join(location).copy_with(fragment=None)


def _read_body(
    response: httpx.Response, max_bytes: int, pdf_max_bytes: int, deadline: float
) -> tuple[Literal["html", "pdf"], bytes]:
    """Read an HTML or PDF response within its size limit and the deadline."""
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    mime = content_type.split(";")[0].strip().lower()
    kind: Literal["html", "pdf"]
    if "html" in mime:
        kind, limit = "html", max_bytes
    elif mime in _PDF_TYPES and pdf_max_bytes > 0:
        kind, limit = "pdf", pdf_max_bytes
    else:
        raise ExtractionError(_content_type_message(mime))
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise FetchError(TOO_LARGE_MESSAGE)  # no need to download what is too large
    body = bytearray()
    for part in response.iter_bytes():
        body.extend(part)
        if len(body) > limit:
            raise FetchError(TOO_LARGE_MESSAGE)
        _check_deadline(deadline)  # a site that sends a few bytes at a time cannot keep the request open
    return kind, bytes(body)


def _status_message(status_code: int) -> str:
    """Describe a failing HTTP status of the requested website in plain words.

    The number is only shown for statuses that have no wording of their own, so that there is something to quote.
    """
    if status_code in {401, 403}:
        return "The website does not allow access to this page. Try a different page."
    if status_code in {404, 410}:
        return "That page was not found. Check the address and try again."
    if status_code == 429:
        return "The website is limiting requests right now. Try again in a moment."
    if status_code >= 500:
        return "The website is having problems right now. Try again later."
    return f"The website could not load this page (error {status_code}). Try a different page."


def _content_type_message(mime: str) -> str:
    """Explain that a link leads to something other than a web page, naming what it is when known."""
    if mime in _PDF_TYPES:
        return PDF_DISABLED_MESSAGE
    kind = _FILE_KINDS.get(mime.split("/")[0], "a file")
    return f"That link is {kind}, not a web page. Use the address of an article or documentation page."


def _host_of(url: str) -> str:
    """The host of an address, for the log: the path and the query of a page can hold private things."""
    return urlsplit(url).hostname or "an unknown host"
