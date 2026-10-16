"""Tests of the safe downloader: no real DNS lookups and no real connections are made."""

import socket
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from chat_with_any_website_agent import fetching
from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError
from chat_with_any_website_agent.fetching import fetch_document, normalize_url, resolve_public_addresses

if TYPE_CHECKING:
    from collections.abc import Callable

PUBLIC = "93.184.216.34"
PAGE_BODY = b"<html>ok</html>"
PDF_BODY = b"%PDF-1.7 minimal"


def _page() -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, content=PAGE_BODY)


def _pdf(content: bytes = PDF_BODY) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/pdf"}, content=content)


def _redirect_to(location: str) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(302, headers={"location": location})


def _answers(monkeypatch: pytest.MonkeyPatch, by_host: dict[str, list[str]]) -> list[str]:
    """Make each name resolve to the listed addresses, in order, and record every lookup."""
    lookups: list[str] = []

    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        lookups.append(host)
        addresses = by_host[host]
        family = socket.AF_INET6 if ":" in addresses[0] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, 0)) for address in addresses]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return lookups


def _fake_dns(*addresses: str) -> Any:
    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in addresses]

    return getaddrinfo


def _network(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    """Make the downloader talk to ``handler``; returns the requests it received."""
    received: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return handler(request)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(record), **kwargs))
    return received


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Answer every request with ``handler``, with a public address for every host."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(PUBLIC))
    _network(monkeypatch, handler)


# --- the address ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "https://example.com/"),
        ("  https://Example.COM/Path?q=1#frag ", "https://example.com/Path?q=1"),
        ("http://example.com", "http://example.com/"),
        ("https://example.com:443/a", "https://example.com:443/a"),
        ("http://example.com:80/", "http://example.com:80/"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw", ["ftp://example.com", "javascript:alert(1)", "https://", "", "http://host:notaport/"])
def test_invalid_addresses_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidURLError) as caught:
        normalize_url(raw)
    assert str(caught.value) == fetching.INVALID_URL_MESSAGE


@pytest.mark.parametrize("raw", ["https://example.com:8443/", "http://example.com:22/", "https://example.com:0/"])
def test_ports_other_than_80_and_443_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidURLError) as caught:
        normalize_url(raw)
    assert str(caught.value) == fetching.DISALLOWED_PORT_MESSAGE


@pytest.mark.parametrize(
    "raw", ["https://user:secret@example.com/", "https://user@example.com/", "https://:pw@example.com/"]
)
def test_addresses_with_a_user_name_or_password_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidURLError) as caught:
        normalize_url(raw)
    assert str(caught.value) == fetching.CREDENTIALS_MESSAGE
    assert "secret" not in str(caught.value)


# --- which addresses are public ---

PRIVATE_FORMS = [
    "127.0.0.1",
    "10.0.0.5",
    "192.168.1.1",
    "172.16.0.9",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "::1",
    "::",
    "fe80::1",
    "fc00::1",
    "::ffff:10.0.0.1",  # IPv4 inside IPv6
    "::ffff:127.0.0.1",
    "64:ff9b::a00:1",  # 10.0.0.1 through a NAT64 gateway
    "64:ff9b::7f00:1",  # 127.0.0.1 through a NAT64 gateway
    "64:ff9b::a9fe:a9fe",  # 169.254.169.254 through a NAT64 gateway
    "2002:a00:1::",  # 10.0.0.1 inside 6to4
]
PUBLIC_FORMS = ["93.184.216.34", "8.8.8.8", "2606:4700::1111", "64:ff9b::808:808", "::ffff:8.8.8.8"]


@pytest.mark.parametrize("address", PRIVATE_FORMS)
def test_private_addresses_are_blocked(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(address))
    with pytest.raises(InvalidURLError) as caught:
        resolve_public_addresses("internal.example")
    assert str(caught.value) == fetching.PRIVATE_ADDRESS_MESSAGE


@pytest.mark.parametrize("address", PUBLIC_FORMS)
def test_public_addresses_are_allowed(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(address))
    assert resolve_public_addresses("example.com") == [address]


def test_a_host_with_any_private_address_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(PUBLIC, "10.0.0.1"))
    with pytest.raises(InvalidURLError):
        resolve_public_addresses("mixed.example")


def test_repeated_addresses_are_listed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(PUBLIC, PUBLIC))
    assert resolve_public_addresses("example.com") == [PUBLIC]


def test_an_unresolvable_host_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    with pytest.raises(InvalidURLError, match=r"No website was found at nope\.invalid"):
        resolve_public_addresses("nope.invalid")


# --- connecting to the address that was checked ---


def test_the_connection_uses_the_address_that_was_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = _answers(monkeypatch, {"example.com": [PUBLIC]})
    received = _network(monkeypatch, lambda request: _page())

    document = fetch_document("https://example.com/a?b=1", timeout=5, max_bytes=1000)

    assert document.body == PAGE_BODY
    assert lookups == ["example.com"]  # looked up once; the connection does not look the name up again
    assert received[0].url.host == PUBLIC
    assert received[0].headers["host"] == "example.com"
    assert received[0].extensions["sni_hostname"] == "example.com"  # the certificate is checked for the name
    assert received[0].url.path == "/a"
    assert received[0].url.query == b"b=1"


def test_a_name_that_changes_its_answer_cannot_reach_a_private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS rebinding: a public answer for the check and a private one for the connection."""
    answers = iter([[PUBLIC], ["127.0.0.1"]])

    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in next(answers)]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    received = _network(monkeypatch, lambda request: _page())

    fetch_document("http://attacker.example/", timeout=5, max_bytes=1000)

    assert [request.url.host for request in received] == [PUBLIC]


def test_ipv6_addresses_are_connected_to_with_the_name_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"v6.example": ["2606:4700::1111"]})
    received = _network(monkeypatch, lambda request: _page())

    fetch_document("https://v6.example/", timeout=5, max_bytes=1000)

    assert received[0].url.host == "2606:4700::1111"
    assert received[0].headers["host"] == "v6.example"


def test_the_next_address_is_tried_when_one_cannot_be_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"two.example": ["2606:4700::1111", PUBLIC]})

    def handler(request: httpx.Request) -> httpx.Response:
        if ":" in request.url.host:
            raise httpx.ConnectError("no route", request=request)
        return _page()

    received = _network(monkeypatch, handler)

    fetch_document("https://two.example/", timeout=5, max_bytes=1000)

    assert [request.url.host for request in received] == ["2606:4700::1111", PUBLIC]


def test_proxy_settings_of_the_environment_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        created.append(kwargs)
        return real_client(transport=httpx.MockTransport(lambda request: _page()), **kwargs)

    _answers(monkeypatch, {"example.com": [PUBLIC]})
    monkeypatch.setattr(httpx, "Client", client)

    fetch_document("https://example.com/", timeout=5, max_bytes=1000)

    assert created[0]["trust_env"] is False


def test_the_user_agent_names_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"example.com": [PUBLIC]})
    received = _network(monkeypatch, lambda request: _page())

    fetch_document("https://example.com/", timeout=5, max_bytes=1000)

    assert received[0].headers["user-agent"] == fetching.USER_AGENT
    assert fetching.PROJECT_URL in fetching.USER_AGENT


# --- redirects ---


def test_a_redirect_is_followed_and_checked_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = _answers(monkeypatch, {"old.example": [PUBLIC], "new.example": ["93.184.216.35"]})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "old.example":
            return httpx.Response(301, headers={"location": "https://new.example/moved#part"})
        return _page()

    received = _network(monkeypatch, handler)

    document = fetch_document("https://old.example/", timeout=5, max_bytes=1000)

    assert [(r.url.host, r.headers["host"], r.url.path) for r in received] == [
        (PUBLIC, "old.example", "/"),
        ("93.184.216.35", "new.example", "/moved"),
    ]
    assert lookups == ["old.example", "new.example"]
    assert document.url == "https://new.example/moved"  # the final address, without the fragment


def test_a_relative_redirect_stays_on_the_same_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"site.example": [PUBLIC]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"}) if request.url.path == "/" else _page()

    received = _network(monkeypatch, handler)

    fetch_document("https://site.example/", timeout=5, max_bytes=1000)

    assert [(r.headers["host"], r.url.path) for r in received] == [("site.example", "/"), ("site.example", "/next")]


@pytest.mark.parametrize(
    ("target", "resolves_to"),
    [
        ("http://127.0.0.1/admin", "127.0.0.1"),
        ("http://localhost/", "127.0.0.1"),
        ("http://[::1]/", "::1"),
        ("http://10.0.0.1/", "10.0.0.1"),
        ("http://metadata.example/", "169.254.169.254"),
        ("http://[64:ff9b::a00:1]/", "64:ff9b::a00:1"),
    ],
)
def test_a_redirect_to_a_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch, target: str, resolves_to: str
) -> None:
    host = httpx.URL(target).host
    _answers(monkeypatch, {"start.example": [PUBLIC], host: [resolves_to]})
    received = _network(monkeypatch, _redirect_to(target))

    with pytest.raises(InvalidURLError) as caught:
        fetch_document("https://start.example/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.PRIVATE_ADDRESS_MESSAGE
    assert [request.headers["host"] for request in received] == ["start.example"]  # the private address got nothing


def test_a_redirect_to_another_protocol_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"start.example": [PUBLIC]})
    _network(monkeypatch, _redirect_to("file:///etc/passwd"))

    with pytest.raises(FetchError) as caught:
        fetch_document("https://start.example/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.UNREACHABLE_MESSAGE


def test_a_redirect_to_another_port_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"start.example": [PUBLIC], "other.example": [PUBLIC]})
    received = _network(monkeypatch, _redirect_to("http://other.example:8080/"))

    with pytest.raises(InvalidURLError) as caught:
        fetch_document("https://start.example/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.DISALLOWED_PORT_MESSAGE
    assert [request.headers["host"] for request in received] == ["start.example"]


def test_a_redirect_to_an_address_with_a_password_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"start.example": [PUBLIC], "other.example": [PUBLIC]})
    _network(monkeypatch, _redirect_to("https://user:secret@other.example/"))

    with pytest.raises(InvalidURLError) as caught:
        fetch_document("https://start.example/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.CREDENTIALS_MESSAGE


def test_the_first_address_is_checked_for_the_port_before_any_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = _answers(monkeypatch, {"example.com": [PUBLIC]})

    with pytest.raises(InvalidURLError) as caught:
        fetch_document("https://example.com:8443/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.DISALLOWED_PORT_MESSAGE
    assert lookups == []


def test_too_many_redirects_end_with_a_plain_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"loop.example": [PUBLIC]})
    received = _network(monkeypatch, _redirect_to("/again"))

    with pytest.raises(FetchError) as caught:
        fetch_document("https://loop.example/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.TOO_MANY_REDIRECTS_MESSAGE
    assert len(received) == fetching.MAX_REDIRECTS + 1


# --- what is accepted, and the limits ---


def test_a_pdf_file_is_returned_when_pdf_support_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, lambda request: _pdf())

    document = fetch_document("https://example.com/paper.pdf", timeout=5, max_bytes=1000, pdf_max_bytes=1000)

    assert (document.kind, document.body, document.url) == ("pdf", PDF_BODY, "https://example.com/paper.pdf")


def test_a_pdf_file_is_refused_when_pdf_support_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, lambda request: _pdf())

    with pytest.raises(ExtractionError) as caught:
        fetch_document("https://example.com/paper.pdf", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.PDF_DISABLED_MESSAGE


def test_pdf_files_are_only_asked_for_when_pdf_support_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"example.com": [PUBLIC]})
    received = _network(monkeypatch, lambda request: _page())

    fetch_document("https://example.com/", timeout=5, max_bytes=1000)
    fetch_document("https://example.com/", timeout=5, max_bytes=1000, pdf_max_bytes=1000)

    assert "application/pdf" not in received[0].headers["accept"]
    assert "application/pdf" in received[1].headers["accept"]


def test_a_pdf_file_has_its_own_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, lambda request: _pdf(b"x" * 50))

    with pytest.raises(FetchError) as caught:
        fetch_document("https://example.com/big.pdf", timeout=5, max_bytes=1000, pdf_max_bytes=10)

    assert str(caught.value) == fetching.TOO_LARGE_MESSAGE


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("image/png", "That link is an image, not a web page."),
        ("video/mp4", "That link is a video, not a web page."),
        ("audio/mpeg", "That link is an audio file, not a web page."),
        ("application/json; charset=utf-8", "That link is a file, not a web page."),
        ("", "That link is a file, not a web page."),
    ],
)
def test_other_file_types_are_named(monkeypatch: pytest.MonkeyPatch, content_type: str, expected: str) -> None:
    _serve(monkeypatch, lambda request: httpx.Response(200, headers={"content-type": content_type}))

    with pytest.raises(ExtractionError) as caught:
        fetch_document("https://example.com/file", timeout=5, max_bytes=1000)

    assert str(caught.value).startswith(expected)
    assert "image/png" not in str(caught.value)  # no technical content types in the message


def test_a_page_that_is_too_large_is_refused_while_it_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The size is not declared, so it is only known once the pieces that arrive add up to more than the limit."""
    headers = {"content-type": "text/html"}
    _serve(monkeypatch, lambda request: httpx.Response(200, headers=headers, content=iter([b"x" * 5] * 10)))

    with pytest.raises(FetchError) as caught:
        fetch_document("https://example.com/", timeout=5, max_bytes=10)

    assert str(caught.value) == fetching.TOO_LARGE_MESSAGE


def test_a_page_that_declares_it_is_too_large_is_refused_without_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[bytes] = []

    def body() -> Any:
        sent.append(b"x")
        yield b"x"

    headers = {"content-type": "text/html", "content-length": "5000"}
    _serve(monkeypatch, lambda request: httpx.Response(200, headers=headers, content=body()))

    with pytest.raises(FetchError) as caught:
        fetch_document("https://example.com/", timeout=5, max_bytes=10)

    assert str(caught.value) == fetching.TOO_LARGE_MESSAGE
    assert sent == []


def test_a_site_that_sends_a_few_bytes_at_a_time_is_stopped_by_the_total_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"slow.example": [PUBLIC]})
    clock = iter(range(0, 1000, 5))  # every look at the clock is five seconds later
    monkeypatch.setattr(fetching, "time", SimpleNamespace(monotonic=lambda: float(next(clock))))
    _network(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=iter([b"a"] * 100)),
    )

    with pytest.raises(FetchError) as caught:
        fetch_document("https://slow.example/", timeout=5, max_bytes=1_000_000)

    assert str(caught.value) == fetching.TIMEOUT_MESSAGE


# --- failures are explained in plain words ---


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, "That page was not found. Check the address and try again."),
        (410, "That page was not found. Check the address and try again."),
        (403, "The website does not allow access to this page. Try a different page."),
        (429, "The website is limiting requests right now. Try again in a moment."),
        (503, "The website is having problems right now. Try again later."),
        (418, "The website could not load this page (error 418). Try a different page."),
    ],
)
def test_http_errors_are_explained_in_plain_words(monkeypatch: pytest.MonkeyPatch, status: int, expected: str) -> None:
    _serve(monkeypatch, lambda request: httpx.Response(status))

    with pytest.raises(FetchError) as caught:
        fetch_document("https://example.com/", timeout=5, max_bytes=1000)

    assert str(caught.value) == expected


def test_a_slow_website_gets_a_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    _serve(monkeypatch, handler)

    with pytest.raises(FetchError) as caught:
        fetch_document("https://example.com/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.TIMEOUT_MESSAGE


def test_a_connection_failure_hides_the_library_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] connection refused to 93.184.216.34", request=request)

    _serve(monkeypatch, handler)

    with pytest.raises(FetchError) as caught:
        fetch_document("https://example.com/", timeout=5, max_bytes=1000)

    assert str(caught.value) == fetching.UNREACHABLE_MESSAGE
    assert "Errno" not in str(caught.value)
