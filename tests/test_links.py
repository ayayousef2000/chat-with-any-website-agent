"""Tests of the choice of the links that the agent may follow, and of how they are kept."""

import pytest

from chat_with_any_website_agent.links import (
    MAX_LINKS,
    MAX_TEXT_CHARS,
    Link,
    collect_links,
    links_from_json,
    links_to_json,
    site_of,
)

PAGE = "https://docs.example/guide/start"


def _urls(*hrefs: str, page: str = PAGE) -> list[str]:
    return [link.url for link in collect_links([(href, "") for href in hrefs], page)]


# --- the site of an address ---


@pytest.mark.parametrize(
    ("url", "site"),
    [
        ("https://docs.example/a", "docs.example"),
        ("https://DOCS.Example:443/a", "docs.example"),
        ("https://www.docs.example/a", "docs.example"),
        ("not an address", ""),
    ],
)
def test_the_site_is_the_host_in_lower_case_without_www(url: str, site: str) -> None:
    assert site_of(url) == site


# --- which links are kept ---


def test_relative_and_absolute_links_to_the_same_site_are_kept_in_the_order_of_the_page() -> None:
    assert _urls("/pricing", "next", "https://docs.example/blog?page=2", "../up") == [
        "https://docs.example/pricing",
        "https://docs.example/guide/next",
        "https://docs.example/blog?page=2",
        "https://docs.example/up",
    ]


def test_www_makes_no_difference_to_the_site() -> None:
    assert _urls("https://www.docs.example/a", page="https://docs.example/") == ["https://www.docs.example/a"]
    assert _urls("https://docs.example/a", page="https://www.docs.example/") == ["https://docs.example/a"]


@pytest.mark.parametrize(
    "href",
    [
        "https://other.example/page",
        "https://docs.example.evil.example/page",  # a longer host that only begins like the site
        "https://sub.docs.example/page",  # another host, which can be another owner
        "mailto:someone@docs.example",
        "tel:+442079460958",
        "javascript:void(0)",
        "ftp://docs.example/file",
        "data:text/html,<p>hi</p>",
        "https://user:secret@docs.example/page",
        "https://docs.example:8443/page",
        "https://docs.example:notaport/page",
        "#top",
        "",
        "   ",
    ],
)
def test_links_that_may_not_be_followed_are_left_out(href: str) -> None:
    assert _urls(href) == []


@pytest.mark.parametrize(
    "href",
    [
        "/logo.PNG",
        "/photo.jpg",
        "/song.mp3",
        "/clip.mp4",
        "/data.zip",
        "/setup.exe",
        "/style.css",
        "/app.js",
        "/a.docx",
    ],
)
def test_links_to_files_that_are_not_read_are_left_out(href: str) -> None:
    assert _urls(href) == []


def test_a_link_to_a_pdf_file_is_kept() -> None:
    assert _urls("/manual.pdf") == ["https://docs.example/manual.pdf"]


def test_the_page_itself_and_a_part_of_it_are_not_links() -> None:
    assert _urls(PAGE, "/guide/start#install", "start", "https://DOCS.example/guide/start") == []


def test_a_fragment_is_dropped_so_two_parts_of_a_page_are_one_link() -> None:
    assert _urls("/api#one", "/api#two", "/api") == ["https://docs.example/api"]


def test_the_host_is_lower_case_and_an_empty_path_is_a_slash() -> None:
    assert _urls("https://DOCS.Example") == ["https://docs.example/"]


def test_a_query_is_part_of_the_address() -> None:
    assert _urls("/search?q=a", "/search?q=b") == [
        "https://docs.example/search?q=a",
        "https://docs.example/search?q=b",
    ]


def test_an_address_is_kept_once_with_the_first_text_that_is_not_empty() -> None:
    anchors = [("/a", ""), ("/b", "Bee"), ("/a", "Ay"), ("/a", "Again"), ("/b", "Other")]

    assert collect_links(anchors, PAGE) == (
        Link("https://docs.example/a", "Ay"),
        Link("https://docs.example/b", "Bee"),
    )


def test_the_text_of_a_link_has_single_spaces_and_a_limit() -> None:
    long = "word " * 100

    [first, second] = collect_links([("/a", "  Getting \n  started  "), ("/b", long)], PAGE)

    assert first.text == "Getting started"
    assert len(second.text) <= MAX_TEXT_CHARS


def test_only_the_first_links_of_a_page_are_kept() -> None:
    anchors = [(f"/page/{number}", "") for number in range(MAX_LINKS + 50)]

    links = collect_links(anchors, PAGE)

    assert len(links) == MAX_LINKS
    assert links[0].url == "https://docs.example/page/0"
    assert links[-1].url == f"https://docs.example/page/{MAX_LINKS - 1}"


def test_a_page_with_no_anchors_has_no_links() -> None:
    assert collect_links([], PAGE) == ()


# --- keeping the links as text ---


def test_links_survive_being_written_and_read() -> None:
    links = (
        Link("https://docs.example/a", "Ay"),
        Link("https://docs.example/b?x=1&y=2", ""),
        Link("https://docs.example/ü", "عربي"),
    )

    assert links_from_json(links_to_json(links)) == links


def test_no_links_are_written_as_an_empty_list() -> None:
    assert links_to_json([]) == "[]"
    assert links_from_json("[]") == ()


@pytest.mark.parametrize(
    "stored",
    [None, "", "not json", "{}", '{"a": 1}', "5", "null", '"text"'],
)
def test_text_that_is_not_a_list_gives_no_links(stored: str | None) -> None:
    assert links_from_json(stored) == ()


def test_entries_that_are_not_an_address_and_a_text_are_skipped() -> None:
    stored = (
        '[["https://docs.example/a","Ay"],["",""],["https://docs.example/b"],[1,"x"],"y",["https://docs.example/c",2]]'
    )

    assert links_from_json(stored) == (Link("https://docs.example/a", "Ay"),)
