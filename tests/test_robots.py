"""Tests of the ``robots.txt`` rules, with a fake download: no website is contacted."""

import logging

import pytest

from chat_with_any_website_agent import robots
from chat_with_any_website_agent.errors import FetchError, InvalidURLError
from chat_with_any_website_agent.fetching import FetchedText
from chat_with_any_website_agent.robots import MAX_CRAWL_DELAY, MAX_ROBOTS_BYTES, USER_AGENT_PRODUCT, RobotsChecker

SITE = "https://docs.example"
ROBOTS_URL = f"{SITE}/robots.txt"


class FakeFile:
    """Answers every request for a file of rules with one answer and remembers the addresses asked for."""

    def __init__(self, status: int = 200, body: str = "", error: BaseException | None = None) -> None:
        self.status = status
        self.body = body
        self.error = error
        self.asked: list[str] = []

    def __call__(self, url: str) -> FetchedText:
        self.asked.append(url)
        if self.error is not None:
            raise self.error
        return FetchedText(status=self.status, body=self.body, url=url)


class Clock:
    """A clock that moves only when it is told to, and a sleep that moves it."""

    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _checker(body: str = "", status: int = 200, error: BaseException | None = None) -> tuple[RobotsChecker, FakeFile]:
    fake = FakeFile(status, body, error)
    return RobotsChecker(fake), fake


# --- the rules in the file ---


def test_a_page_is_allowed_when_the_file_forbids_nothing() -> None:
    checker, _ = _checker("User-agent: *\nDisallow:\n")

    assert checker.allows(f"{SITE}/anything")


def test_a_forbidden_path_is_not_allowed_and_the_rest_is() -> None:
    checker, _ = _checker("User-agent: *\nDisallow: /private\n")

    assert not checker.allows(f"{SITE}/private/page")
    assert not checker.allows(f"{SITE}/private")
    assert checker.allows(f"{SITE}/public/page")


def test_the_whole_site_can_be_forbidden() -> None:
    checker, _ = _checker("User-agent: *\nDisallow: /\n")

    assert not checker.allows(f"{SITE}/")
    assert not checker.allows(f"{SITE}/a")


def test_an_allow_rule_can_open_a_part_of_a_forbidden_path() -> None:
    checker, _ = _checker("User-agent: *\nDisallow: /private\nAllow: /private/open\n")

    assert checker.allows(f"{SITE}/private/open/page")
    assert not checker.allows(f"{SITE}/private/closed")


def test_wildcards_and_the_end_of_an_address_are_understood() -> None:
    checker, _ = _checker("User-agent: *\nDisallow: /*?session=\nDisallow: /*.json$\n")

    assert not checker.allows(f"{SITE}/page?session=1")
    assert not checker.allows(f"{SITE}/data/list.json")
    assert checker.allows(f"{SITE}/data/list.json.html")
    assert checker.allows(f"{SITE}/page?id=1")


def test_the_rules_for_this_program_come_before_the_rules_for_everyone() -> None:
    body = f"User-agent: *\nDisallow:\n\nUser-agent: {USER_AGENT_PRODUCT}\nDisallow: /drafts\n"
    checker, _ = _checker(body)

    assert not checker.allows(f"{SITE}/drafts/one")
    assert checker.allows(f"{SITE}/final/one")


def test_the_rules_for_another_program_do_not_apply() -> None:
    checker, _ = _checker("User-agent: SomeOtherBot\nDisallow: /\n")

    assert checker.allows(f"{SITE}/a")


def test_comments_and_unknown_lines_are_ignored() -> None:
    body = "# rules\nSitemap: https://docs.example/sitemap.xml\nUser-agent: *\nDisallow: /x # no\nFoo: bar\n"
    checker, _ = _checker(body)

    assert not checker.allows(f"{SITE}/x")
    assert checker.allows(f"{SITE}/y")


@pytest.mark.parametrize(
    "body", ["", "\x00\x00\x00", "Disallow: /", "User-agent: *\nDisallow", "a" * 5000, chr(0xFFFD) * 2]
)
def test_a_file_that_makes_no_sense_does_not_stop_the_agent(body: str) -> None:
    checker, _ = _checker(body)

    assert checker.allows(f"{SITE}/page")  # without a rule that applies, nothing is forbidden


def test_a_very_large_file_is_handled() -> None:
    body = "User-agent: *\n" + "Disallow: /path/*/*/*/*/*/*/*/x\n" * 5000
    checker, _ = _checker(body[:MAX_ROBOTS_BYTES])

    assert checker.allows(f"{SITE}/a/b/c")


# --- what the answer of the site means ---


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 451])
def test_a_file_that_is_not_there_or_not_given_means_nothing_is_forbidden(status: int) -> None:
    checker, _ = _checker("User-agent: *\nDisallow: /\n", status)  # whatever the body says, it was not the file

    assert checker.allows(f"{SITE}/page")


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_site_in_trouble_or_asking_to_slow_down_means_nothing_is_allowed(status: int) -> None:
    checker, _ = _checker("User-agent: *\nDisallow:\n", status)

    assert not checker.allows(f"{SITE}/page")


def test_a_redirect_status_that_was_not_followed_means_nothing_is_allowed() -> None:
    checker, _ = _checker("", 302)

    assert not checker.allows(f"{SITE}/page")


@pytest.mark.parametrize(
    "error",
    [
        FetchError("The website could not be reached."),
        FetchError("The website took too long to respond."),
        InvalidURLError("That address is on a private network."),
    ],
)
def test_a_site_that_cannot_be_asked_means_nothing_is_allowed(error: BaseException) -> None:
    checker, _ = _checker(error=error)

    assert not checker.allows(f"{SITE}/page")


def test_the_reason_is_logged_with_the_site_and_not_the_page(caplog: pytest.LogCaptureFixture) -> None:
    checker, _ = _checker(status=503)

    with caplog.at_level(logging.INFO, logger="chat_with_any_website_agent.robots"):
        checker.allows(f"{SITE}/a/secret/page?token=abc")

    assert "docs.example" in caplog.text
    assert "secret" not in caplog.text
    assert "token" not in caplog.text


# --- asking once ---


def test_the_file_is_asked_for_once_for_a_site() -> None:
    checker, fake = _checker("User-agent: *\nDisallow: /private\n")

    for path in ("/a", "/b", "/private/c", "/a"):
        checker.allows(f"{SITE}{path}")
    checker.crawl_delay(f"{SITE}/a")

    assert fake.asked == [ROBOTS_URL]


def test_a_failure_is_not_asked_again_either() -> None:
    checker, fake = _checker(error=FetchError("The website could not be reached."))

    assert not checker.allows(f"{SITE}/a")
    assert not checker.allows(f"{SITE}/b")
    assert len(fake.asked) == 1


def test_each_site_has_its_own_file() -> None:
    checker, fake = _checker("User-agent: *\nDisallow:\n")

    checker.allows("https://docs.example/a")
    checker.allows("https://docs.example:443/a")
    checker.allows("http://docs.example/a")  # another scheme is another origin
    checker.allows("https://www.docs.example/a")

    assert fake.asked == [
        "https://docs.example/robots.txt",
        "https://docs.example:443/robots.txt",
        "http://docs.example/robots.txt",
        "https://www.docs.example/robots.txt",
    ]


def test_the_name_of_the_site_is_read_in_lower_case() -> None:
    checker, fake = _checker()

    checker.allows("https://DOCS.Example/a")
    checker.allows("https://docs.example/b")

    assert fake.asked == [ROBOTS_URL]


@pytest.mark.parametrize(
    "url", ["", "not an address", "ftp://docs.example/file", "mailto:a@docs.example", "https:///x"]
)
def test_something_that_is_not_a_web_address_is_not_allowed_and_nothing_is_asked(url: str) -> None:
    checker, fake = _checker()

    assert not checker.allows(url)
    assert checker.crawl_delay(url) == 0.0
    checker.wait_turn(url)
    assert fake.asked == []


# --- the pause between requests ---


def test_the_pause_a_site_asks_for_is_read() -> None:
    checker, _ = _checker("User-agent: *\nCrawl-delay: 3\n")

    assert checker.crawl_delay(f"{SITE}/a") == 3.0


@pytest.mark.parametrize(
    ("body", "status", "expected"),
    [
        ("User-agent: *\nDisallow: /x\n", 200, 0.0),  # none asked for
        ("User-agent: *\nCrawl-delay: 600\n", 200, MAX_CRAWL_DELAY),  # too long is cut
        ("User-agent: *\nCrawl-delay: 0.5\n", 200, 0.5),
        ("User-agent: *\nCrawl-delay: soon\n", 200, 0.0),
        ("User-agent: *\nCrawl-delay: 3\n", 404, 0.0),  # there is no file
        ("User-agent: *\nCrawl-delay: 3\n", 503, 0.0),  # nothing is opened anyway
    ],
)
def test_the_pause_is_limited_and_zero_when_unknown(body: str, status: int, expected: float) -> None:
    checker, _ = _checker(body, status)

    assert checker.crawl_delay(f"{SITE}/a") == expected


def _polite(body: str) -> tuple[RobotsChecker, Clock]:
    clock = Clock()
    checker = RobotsChecker(FakeFile(200, body), sleep=clock.sleep, clock=clock)
    return checker, clock


def test_the_first_request_to_a_site_does_not_wait() -> None:
    checker, clock = _polite("User-agent: *\nCrawl-delay: 4\n")

    checker.wait_turn(f"{SITE}/a")

    assert clock.slept == []


def test_the_next_request_waits_for_the_rest_of_the_pause() -> None:
    checker, clock = _polite("User-agent: *\nCrawl-delay: 4\n")

    checker.wait_turn(f"{SITE}/a")
    clock.now += 1.5  # some of the pause passed while the page was read
    checker.wait_turn(f"{SITE}/b")

    assert clock.slept == [2.5]


def test_a_request_after_the_pause_does_not_wait() -> None:
    checker, clock = _polite("User-agent: *\nCrawl-delay: 4\n")

    checker.wait_turn(f"{SITE}/a")
    clock.now += 4.0
    checker.wait_turn(f"{SITE}/b")

    assert clock.slept == []


def test_a_site_that_asks_for_no_pause_is_never_waited_for() -> None:
    checker, clock = _polite("User-agent: *\nDisallow: /x\n")

    checker.wait_turn(f"{SITE}/a")
    checker.wait_turn(f"{SITE}/b")

    assert clock.slept == []


def test_the_pause_is_kept_for_each_site_on_its_own() -> None:
    clock = Clock()
    checker = RobotsChecker(FakeFile(200, "User-agent: *\nCrawl-delay: 4\n"), sleep=clock.sleep, clock=clock)

    checker.wait_turn("https://docs.example/a")
    checker.wait_turn("https://other.example/a")

    assert clock.slept == []


# --- the default download ---


def test_by_default_the_file_is_downloaded_with_the_safe_download(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float, int]] = []

    def fake_fetch_text(url: str, *, timeout: float, max_bytes: int) -> FetchedText:
        calls.append((url, timeout, max_bytes))
        return FetchedText(status=200, body="User-agent: *\nDisallow: /no\n", url=url)

    monkeypatch.setattr(robots, "fetch_text", fake_fetch_text)

    checker = RobotsChecker(timeout=7.0)

    assert not checker.allows("https://docs.example/no")
    assert calls == [(ROBOTS_URL, 7.0, MAX_ROBOTS_BYTES)]


def test_an_address_that_cannot_even_be_split_is_not_allowed() -> None:
    checker, fake = _checker()

    assert not checker.allows("http://[not-an-address/")
    assert fake.asked == []
