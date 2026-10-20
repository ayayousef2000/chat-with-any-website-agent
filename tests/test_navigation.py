"""Tests of the opening of links, with fakes instead of the ingestor, the store and the sites."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError, ServiceBusyError
from chat_with_any_website_agent.fetching import FetchedText
from chat_with_any_website_agent.ingest import IngestResult
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.navigation import ROBOTS_REFUSAL, FollowResult, LinkFollower
from chat_with_any_website_agent.robots import RobotsChecker
from chat_with_any_website_agent.vector_store import SourceRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

PAGE = "https://docs.example/tutorial/control"
LINKS = (Link("https://docs.example/a", "A"), Link("https://docs.example/b", ""))
LOADED = IngestResult(url=PAGE, reused=False, title="Control", chunk_count=4)


@dataclass
class FakeStore:
    """Knows the record of some pages."""

    records: Mapping[str, SourceRecord] = field(default_factory=dict)

    def source_record(self, url: str) -> SourceRecord | None:
        return self.records.get(url)


@dataclass
class FakeIngestor:
    """Stores nothing; answers like the real one or fails, and records how it was asked."""

    error: BaseException | None = None
    result: IngestResult = LOADED
    calls: list[tuple[str, bool, str | None]] = field(default_factory=list)

    def ingest(self, url: str, *, check: bool = True, site: str | None = None) -> IngestResult:
        self.calls.append((url, check, site))
        if self.error is not None:
            raise self.error
        return self.result


def _robots(body: str = "", status: int = 200, asked: list[str] | None = None) -> RobotsChecker:
    def fetch(url: str) -> FetchedText:
        if asked is not None:
            asked.append(url)
        return FetchedText(status=status, body=body, url=url)

    return RobotsChecker(fetch)


def _follower(
    ingestor: FakeIngestor | None = None,
    store: FakeStore | None = None,
    robots: RobotsChecker | None = None,
    *,
    check: bool = True,
) -> tuple[LinkFollower, FakeIngestor]:
    ingestor = ingestor or FakeIngestor()
    return LinkFollower(ingestor, store or FakeStore(), robots or _robots(), check=check), ingestor


# --- the links of a stored page ---


def test_the_links_of_a_stored_page_are_read_from_its_record() -> None:
    follower, _ = _follower(store=FakeStore({PAGE: SourceRecord(links=LINKS)}))

    assert follower.links_of(PAGE) == LINKS


@pytest.mark.parametrize("records", [{}, {PAGE: SourceRecord()}])
def test_a_page_that_is_not_stored_or_has_no_links_gives_none(records: dict[str, SourceRecord]) -> None:
    follower, _ = _follower(store=FakeStore(records))

    assert follower.links_of(PAGE) == ()


# --- opening a link ---


def test_a_link_that_the_site_allows_is_loaded_on_its_own_site() -> None:
    follower, ingestor = _follower()

    result = follower.follow(PAGE)

    assert result == FollowResult(url=PAGE, loaded=True, title="Control", chunk_count=4)
    assert ingestor.calls == [(PAGE, True, "docs.example")]


def test_a_stored_page_that_is_reused_is_a_loaded_page_too() -> None:
    follower, _ = _follower(FakeIngestor(result=IngestResult(url=PAGE, reused=True)))

    assert follower.follow(PAGE) == FollowResult(url=PAGE, loaded=True)


def test_the_address_the_page_is_stored_under_is_returned() -> None:
    stored = IngestResult(url="https://docs.example/tutorial/control/", reused=True)
    follower, _ = _follower(FakeIngestor(result=stored))

    assert follower.follow(PAGE).url == "https://docs.example/tutorial/control/"


def test_whether_a_stored_page_is_checked_is_decided_by_the_caller() -> None:
    follower, ingestor = _follower(check=False)

    follower.follow(PAGE)

    assert ingestor.calls[0][1] is False


def test_a_page_that_the_robots_file_forbids_is_not_loaded() -> None:
    follower, ingestor = _follower(robots=_robots("User-agent: *\nDisallow: /tutorial\n"))

    result = follower.follow(PAGE)

    assert result == FollowResult(url=PAGE, loaded=False, note=ROBOTS_REFUSAL)
    assert ingestor.calls == []


def test_a_site_that_cannot_be_asked_for_its_rules_is_not_followed() -> None:
    follower, ingestor = _follower(robots=_robots(status=503))

    assert not follower.follow(PAGE).loaded
    assert ingestor.calls == []


def test_the_robots_file_is_asked_for_once_for_several_links_of_a_site() -> None:
    asked: list[str] = []
    follower, _ = _follower(robots=_robots(asked=asked))

    follower.follow("https://docs.example/a")
    follower.follow("https://docs.example/b")

    assert asked == ["https://docs.example/robots.txt"]


@pytest.mark.parametrize(
    "error",
    [
        FetchError("The website could not be reached."),
        InvalidURLError("That address is on a private network."),
        ExtractionError("No readable text was found on this page."),
    ],
)
def test_a_page_that_cannot_be_read_is_a_result_with_the_reason(error: BaseException) -> None:
    follower, _ = _follower(FakeIngestor(error=error))

    assert follower.follow(PAGE) == FollowResult(url=PAGE, loaded=False, note=str(error))


def test_a_failure_of_a_service_is_not_hidden() -> None:
    follower, _ = _follower(FakeIngestor(error=ServiceBusyError("Cohere is busy.", retry_after=30.0)))

    with pytest.raises(ServiceBusyError):
        follower.follow(PAGE)
