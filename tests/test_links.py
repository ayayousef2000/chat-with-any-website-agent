"""Tests of the choice of the links that the agent may follow, and of how they are kept."""

import pytest

from chat_with_any_website_agent.links import (
    MAX_LINKS,
    MAX_TEXT_CHARS,
    Link,
    collect_links,
    links_from_json,
    links_to_json,
    shortlist_links,
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


# --- the links worth showing to a model ---


def _numbered(count: int) -> tuple[Link, ...]:
    return tuple(Link(f"https://docs.example/page-{number}", f"Page {number}") for number in range(count))


def test_links_that_all_fit_are_returned_as_they_are_in_the_order_of_the_page() -> None:
    links = _numbered(5)

    assert shortlist_links(links, "page 3 statement", limit=5) == links
    assert shortlist_links(links, "", limit=10) == links


def test_the_links_that_share_words_with_the_topic_come_first_when_there_are_too_many() -> None:
    wanted = (
        Link("https://docs.example/tutorial/errors", "Errors and Exceptions"),
        Link("https://docs.example/tutorial/controlflow", "More Control Flow Tools"),
    )
    links = (*_numbered(10), *wanted)

    chosen = shortlist_links(links, "Which kinds of errors does the tutorial distinguish? exceptions", limit=3)

    assert chosen[0] == wanted[0]
    assert len(chosen) == 3


def test_word_endings_are_ignored_when_links_are_compared_with_the_topic() -> None:
    other = _numbered(6)
    plural = Link("https://docs.example/x", "Break and continue Statements")

    chosen = shortlist_links((*other, plural), "What does the break statement do?", limit=2)

    assert chosen[0] == plural


def test_the_words_of_the_address_count_as_well_as_the_text() -> None:
    other = _numbered(6)
    by_address = Link("https://docs.example/tutorial/data_structures", "")

    chosen = shortlist_links((*other, by_address), "data structures", limit=2)

    assert chosen[0] == by_address


def test_links_that_match_equally_keep_the_order_of_the_page() -> None:
    links = _numbered(8)

    assert shortlist_links(links, "nothing in common here", limit=3) == links[:3]


def test_a_limit_of_zero_shows_nothing() -> None:
    assert shortlist_links(_numbered(3), "page", limit=0) == ()


# --- links near the pages the question is about ---

TUTORIAL_START = "https://docs.example/3/tutorial/index.html"
TUTORIAL_CHAPTER = Link("https://docs.example/3/tutorial/errors.html", "Errors")
REFERENCE_INDEX = Link("https://docs.example/3/reference/index.html", "The Language Reference")


def test_links_in_the_same_part_of_the_site_are_listed_first_even_when_all_of_them_fit() -> None:
    links = (REFERENCE_INDEX, TUTORIAL_CHAPTER)

    assert shortlist_links(links, "", limit=5, near=[TUTORIAL_START]) == (TUTORIAL_CHAPTER, REFERENCE_INDEX)


def test_links_keep_the_order_of_the_page_inside_each_group() -> None:
    inside = tuple(Link(f"https://docs.example/3/tutorial/page-{number}.html", "") for number in range(3))
    outside = tuple(Link(f"https://docs.example/3/library/page-{number}.html", "") for number in range(3))

    shortlisted = shortlist_links(
        (outside[0], inside[0], outside[1], inside[1], outside[2], inside[2]), "", 9, [TUTORIAL_START]
    )

    assert shortlisted == (*inside, *outside)


def test_a_directory_inside_the_part_of_the_page_is_near_it() -> None:
    deeper = Link("https://docs.example/3/tutorial/sub/page.html", "")

    assert shortlist_links((REFERENCE_INDEX, deeper), "", 5, [TUTORIAL_START]) == (deeper, REFERENCE_INDEX)


def test_a_directory_whose_name_only_starts_the_same_way_is_not_near() -> None:
    lookalike = Link("https://docs.example/3/tutorial-extras/page.html", "")

    assert shortlist_links((lookalike, TUTORIAL_CHAPTER), "", 5, [TUTORIAL_START]) == (TUTORIAL_CHAPTER, lookalike)


@pytest.mark.parametrize("page", ["https://docs.example/", "https://docs.example/index.html", "https://docs.example"])
def test_a_page_at_the_top_of_the_site_prefers_no_link(page: str) -> None:
    links = (REFERENCE_INDEX, TUTORIAL_CHAPTER)

    assert shortlist_links(links, "", limit=5, near=[page]) == links


def test_a_link_to_another_site_is_not_near_a_page_with_the_same_path() -> None:
    elsewhere = Link("https://other.example/3/tutorial/errors.html", "")

    assert shortlist_links((elsewhere, TUTORIAL_CHAPTER), "", 5, [TUTORIAL_START]) == (TUTORIAL_CHAPTER, elsewhere)


def test_the_site_is_compared_without_a_leading_www() -> None:
    page = "https://www.docs.example/3/tutorial/index.html"

    assert shortlist_links((REFERENCE_INDEX, TUTORIAL_CHAPTER), "", 5, [page]) == (TUTORIAL_CHAPTER, REFERENCE_INDEX)


def test_several_pages_make_each_of_their_parts_near() -> None:
    library = Link("https://docs.example/3/library/os.html", "")
    other = Link("https://docs.example/3/faq/index.html", "")
    pages = [TUTORIAL_START, "https://docs.example/3/library/index.html"]

    assert shortlist_links((other, library, TUTORIAL_CHAPTER), "", 5, pages) == (library, TUTORIAL_CHAPTER, other)


def test_words_in_common_with_the_topic_still_decide_which_links_are_kept() -> None:
    wanted = Link("https://docs.example/3/library/stdtypes.html", "Built-in Types and append")
    near = tuple(Link(f"https://docs.example/3/tutorial/page-{number}.html", "") for number in range(6))

    shortlisted = shortlist_links((*near, wanted), "What does append do?", 2, [TUTORIAL_START])

    assert wanted in shortlisted
    assert shortlisted == (near[0], wanted)  # the near link that is kept is listed before it


def test_among_links_that_match_equally_the_near_ones_are_kept() -> None:
    far = tuple(Link(f"https://docs.example/3/library/page-{number}.html", "") for number in range(5))

    shortlisted = shortlist_links((*far, TUTORIAL_CHAPTER), "nothing in common", 2, [TUTORIAL_START])

    assert shortlisted == (TUTORIAL_CHAPTER, far[0])


def test_no_pages_near_change_nothing() -> None:
    links = (REFERENCE_INDEX, TUTORIAL_CHAPTER)

    assert shortlist_links(links, "", limit=5) == links
    assert shortlist_links(links, "", limit=5, near=[]) == links
