"""Tests of the crawl of a site, with a fake link follower: no website is contacted."""

from dataclasses import dataclass, field

from chat_with_any_website_agent.crawl import CrawlLimits, crawl_site
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.navigation import FollowResult

ROOT = "https://docs.example/"


def _url(name: str) -> str:
    return f"https://docs.example/{name}"


def _links(*names: str) -> tuple[Link, ...]:
    return tuple(Link(_url(name), name) for name in names)


@dataclass
class FakeSite:
    """The links of some pages, and the pages that cannot be opened; records what was opened."""

    links: dict[str, tuple[Link, ...]]
    unopenable: set[str] = field(default_factory=set)
    opened: list[str] = field(default_factory=list)

    def links_of(self, url: str) -> tuple[Link, ...]:
        return self.links.get(url, ())

    def follow(self, url: str) -> FollowResult:
        self.opened.append(url)
        if url in self.unopenable:
            return FollowResult(url=url, loaded=False, note="no")
        return FollowResult(url=url, loaded=True)


SITE = {
    ROOT: _links("a", "b"),
    _url("a"): _links("a1", "a2"),
    _url("b"): _links("b1"),
    _url("a1"): _links("a11"),
}


def test_the_pages_are_loaded_level_by_level_in_the_order_the_pages_list_them() -> None:
    site = FakeSite(SITE)

    loaded = crawl_site(site, ROOT, CrawlLimits(max_pages=10, max_depth=2))

    assert loaded == [_url(name) for name in ("a", "b", "a1", "a2", "b1")]  # a11 is three links away
    assert site.opened == loaded


def test_the_depth_limits_how_far_the_crawl_goes() -> None:
    assert crawl_site(FakeSite(SITE), ROOT, CrawlLimits(max_pages=10, max_depth=1)) == [_url("a"), _url("b")]
    assert crawl_site(FakeSite(SITE), ROOT, CrawlLimits(max_pages=10, max_depth=0)) == []


def test_the_number_of_pages_is_limited_and_no_more_are_opened() -> None:
    site = FakeSite(SITE)

    loaded = crawl_site(site, ROOT, CrawlLimits(max_pages=3, max_depth=5))

    assert loaded == [_url("a"), _url("b"), _url("a1")]
    assert len(site.opened) == 3


def test_a_page_is_opened_once_however_many_pages_link_to_it() -> None:
    site = FakeSite({ROOT: _links("a", "b"), _url("a"): _links("b", "c"), _url("b"): _links("a", "c")})

    loaded = crawl_site(site, ROOT, CrawlLimits(max_pages=10, max_depth=5))

    assert loaded == [_url("a"), _url("b"), _url("c")]
    assert sorted(site.opened) == sorted(loaded)


def test_the_first_page_is_not_loaded_again() -> None:
    site = FakeSite({ROOT: _links("a"), _url("a"): (Link(ROOT, "Home"),)})

    assert crawl_site(site, ROOT, CrawlLimits(max_pages=10, max_depth=5)) == [_url("a")]


def test_a_page_that_cannot_be_opened_is_skipped_and_its_links_are_not_followed() -> None:
    site = FakeSite(SITE, unopenable={_url("a")})

    loaded = crawl_site(site, ROOT, CrawlLimits(max_pages=10, max_depth=2))

    assert loaded == [_url("b"), _url("b1")]


def test_a_page_that_ends_on_a_page_already_loaded_counts_once() -> None:
    class Redirecting(FakeSite):
        def follow(self, url: str) -> FollowResult:
            self.opened.append(url)
            return FollowResult(url=_url("a"), loaded=True)  # every link ends on the same page

    site = Redirecting({ROOT: _links("x", "y")})

    assert crawl_site(site, ROOT, CrawlLimits(max_pages=10, max_depth=2)) == [_url("a")]


def test_a_site_without_links_loads_nothing() -> None:
    assert crawl_site(FakeSite({}), ROOT, CrawlLimits(max_pages=10, max_depth=2)) == []
