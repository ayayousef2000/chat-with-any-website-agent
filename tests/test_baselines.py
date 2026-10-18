"""Tests of the baseline systems, with fakes instead of the services: no website, model or database is contacted."""

from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.documents import Document

from chat_with_any_website_agent import baselines
from chat_with_any_website_agent.baselines import PlainSystem, create_crawl_system, create_plain_system
from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.crawl import CrawlLimits
from chat_with_any_website_agent.errors import ExtractionError, FetchError, ServiceBusyError
from chat_with_any_website_agent.evaluation import Case, Resettable
from chat_with_any_website_agent.graph import NO_ANSWER_REASON, NO_EXCERPTS_REASON
from chat_with_any_website_agent.groq_client import ChatReply
from chat_with_any_website_agent.ingest import IngestResult
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.navigation import FollowResult
from chat_with_any_website_agent.prompts import NOT_COVERED_MESSAGE
from chat_with_any_website_agent.systems import SYSTEMS
from chat_with_any_website_agent.usage import UsageCounter

if TYPE_CHECKING:
    from chat_with_any_website_agent.wiring import AgentRuntime

START = "https://docs.example/tutorial"
OTHER = "https://docs.example/other"
PAGE_A = "https://docs.example/tutorial/a"
PAGE_B = "https://docs.example/tutorial/b"
QUESTION = "What does pass do?"


class FakeIngestor:
    """Stores nothing; it normalizes the address by adding a slash and can fail for some of them."""

    def __init__(self, failures: dict[str, BaseException] | None = None) -> None:
        self.failures = failures or {}
        self.calls: list[tuple[str, bool]] = []

    def ingest(self, url: str, *, check: bool = True) -> IngestResult:
        self.calls.append((url, check))
        if url in self.failures:
            raise self.failures[url]
        return IngestResult(url=f"{url}/", reused=True)


class FakeRetriever:
    """Finds the documents it was given and records the pages it was asked to search."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.urls: list[list[str]] = []
        self.questions: list[str] = []

    def model_copy(self, *, update: dict[str, Any]) -> FakeRetriever:
        self.urls.append(update["urls"])
        return self

    def invoke(self, question: str) -> list[Document]:
        self.questions.append(question)
        return self.documents


class FakeSite:
    """Links of the stored pages and what opening them comes to; records what was opened."""

    def __init__(self, links: dict[str, tuple[Link, ...]]) -> None:
        self.links = links
        self.opened: list[str] = []

    def links_of(self, url: str) -> tuple[Link, ...]:
        return self.links.get(url, ())

    def follow(self, url: str) -> FollowResult:
        self.opened.append(url)
        return FollowResult(url=url, loaded=True)


class FakeClient:
    """Answers with the replies it was given, in order, and records its requests."""

    model = "model-x"

    def __init__(self, *replies: str) -> None:
        self.replies = deque(replies)
        self.requests: list[tuple[list[Any], dict[str, Any]]] = []

    def complete(self, messages: Any, **options: Any) -> ChatReply:
        self.requests.append((list(messages), options))
        return ChatReply(text=self.replies.popleft(), prompt_tokens=10, completion_tokens=5, key_number=1)


def _document(text: str, url: str = f"{START}/") -> Document:
    return Document(page_content=text, metadata={"url": url, "title": "Tutorial", "heading": "", "chunk_index": 0})


def _system(
    reply: str = "The pass statement does nothing [1].",
    documents: list[Document] | None = None,
    *,
    ingestor: FakeIngestor | None = None,
    site: FakeSite | None = None,
    crawl: CrawlLimits | None = None,
) -> tuple[PlainSystem, FakeIngestor, FakeRetriever, FakeClient, FakeSite, list[str]]:
    fake_ingestor, client = ingestor or FakeIngestor(), FakeClient(reply)
    retriever = FakeRetriever([_document("The pass statement does nothing.")] if documents is None else documents)
    fake_site = site or FakeSite({})
    events: list[str] = []
    store = SimpleNamespace(clear=lambda: events.append("cleared"))
    runtime = SimpleNamespace(
        ingestor=fake_ingestor,
        retriever=retriever,
        answer_model=GroqChatModel(client),
        follower=fake_site,
        store=store,
        close=lambda: events.append("closed"),
    )
    return PlainSystem(cast("AgentRuntime", runtime), crawl=crawl), fake_ingestor, retriever, client, fake_site, events


# --- one search and one answer ---


def test_the_question_is_answered_from_one_search_and_one_call_to_the_model() -> None:
    system, ingestor, retriever, client, _, _ = _system()

    outcome = system.ask(Case(url=START, question=QUESTION))

    assert outcome.answer == "The pass statement does nothing [1]."
    assert outcome.excerpts == ("The pass statement does nothing.",)
    assert outcome.sources == (f"{START}/",)
    assert outcome.reason is None
    assert outcome.steps == (f'Searched 1 page(s) for "{QUESTION}" and found 1 excerpt(s).', "Wrote an answer.")
    assert retriever.urls == [[f"{START}/"]]
    assert retriever.questions == [QUESTION]
    assert ingestor.calls == [(START, False)]  # the page was checked when the run prepared it
    assert len(client.requests) == 1
    messages, options = client.requests[0]
    assert options == {"temperature": 0}
    assert "[1] Tutorial\nThe pass statement does nothing." in messages[-1]["content"]


def test_further_sources_of_the_question_are_searched_too() -> None:
    system, ingestor, retriever, _, _, _ = _system()

    system.ask(Case(url=START, question=QUESTION, sources=(OTHER,)))

    assert ingestor.calls == [(START, False), (OTHER, False)]
    assert retriever.urls == [[f"{START}/", f"{OTHER}/"]]


def test_the_sources_are_those_of_the_excerpts_each_once() -> None:
    documents = [_document("one", f"{START}/"), _document("two", PAGE_A), _document("three", f"{START}/")]
    system, *_ = _system(documents=documents)

    assert system.ask(Case(url=START, question=QUESTION)).sources == (f"{START}/", PAGE_A)


def test_a_model_that_says_the_excerpts_hold_no_answer_gives_the_standard_message() -> None:
    system, *_ = _system(reply="NOT_COVERED")

    outcome = system.ask(Case(url=START, question=QUESTION))

    assert outcome.answer == NOT_COVERED_MESSAGE
    assert outcome.reason == NO_ANSWER_REASON
    assert outcome.excerpts == ("The pass statement does nothing.",)
    assert outcome.steps[-1] == "The model found no answer in the excerpts."


def test_a_search_that_finds_nothing_does_not_call_the_model() -> None:
    system, _, _, client, _, _ = _system(documents=[])

    outcome = system.ask(Case(url=START, question=QUESTION))

    assert outcome == type(outcome)(answer=NOT_COVERED_MESSAGE, reason=NO_EXCERPTS_REASON, steps=outcome.steps)
    assert client.requests == []


@pytest.mark.parametrize("error", [FetchError("The website could not be reached."), ExtractionError("No text.")])
def test_a_source_that_cannot_be_loaded_is_the_answer(error: BaseException) -> None:
    system, _, retriever, client, _, _ = _system(ingestor=FakeIngestor({START: error}))

    outcome = system.ask(Case(url=START, question=QUESTION))

    assert outcome.answer == f"The source could not be loaded: {error}"
    assert outcome.reason == str(error)
    assert retriever.questions == []
    assert client.requests == []


def test_a_busy_service_is_not_hidden() -> None:
    system, *_ = _system(ingestor=FakeIngestor({START: ServiceBusyError("Cohere is busy.", retry_after=30.0)}))

    with pytest.raises(ServiceBusyError):
        system.ask(Case(url=START, question=QUESTION))


# --- the crawl ---


def test_without_a_crawl_preparing_loads_the_source_and_nothing_else() -> None:
    system, ingestor, _, _, site, _ = _system(site=FakeSite({f"{START}/": (Link(PAGE_A, "A"),)}))

    system.prepare(START)

    assert ingestor.calls == [(START, True)]
    assert site.opened == []


def test_a_crawl_loads_the_pages_around_the_source_and_every_question_searches_them() -> None:
    site = FakeSite({f"{START}/": (Link(PAGE_A, "A"), Link(PAGE_B, "B"))})
    system, _, retriever, _, _, _ = _system(site=site, crawl=CrawlLimits(max_pages=5, max_depth=1))

    system.prepare(START)
    outcome = system.ask(Case(url=START, question=QUESTION))

    assert site.opened == [PAGE_A, PAGE_B]
    assert retriever.urls == [[f"{START}/", PAGE_A, PAGE_B]]
    assert outcome.steps[0] == f'Searched 3 page(s) for "{QUESTION}" and found 1 excerpt(s).'


def test_the_crawl_is_limited_by_its_limits() -> None:
    site = FakeSite({f"{START}/": (Link(PAGE_A, "A"), Link(PAGE_B, "B"))})
    system, *_ = _system(site=site, crawl=CrawlLimits(max_pages=1, max_depth=1))

    system.prepare(START)

    assert site.opened == [PAGE_A]


def test_a_source_that_cannot_be_prepared_is_only_logged_and_the_crawl_is_skipped() -> None:
    site = FakeSite({f"{START}/": (Link(PAGE_A, "A"),)})
    system, *_ = _system(
        ingestor=FakeIngestor({START: FetchError("The website could not be reached.")}),
        site=site,
        crawl=CrawlLimits(max_pages=5, max_depth=1),
    )

    system.prepare(START)

    assert site.opened == []


def test_the_pages_of_a_crawl_belong_to_the_source_that_was_crawled() -> None:
    site = FakeSite({f"{START}/": (Link(PAGE_A, "A"),)})
    system, _, retriever, *_ = _system(site=site, crawl=CrawlLimits(max_pages=5, max_depth=1))

    system.prepare(START)
    system.ask(Case(url=OTHER, question=QUESTION))  # a question about another source does not search those pages

    assert retriever.urls == [[f"{OTHER}/"]]


# --- emptying and closing ---


def test_the_system_can_empty_the_tenant_and_close() -> None:
    system, *_, events = _system()

    assert isinstance(system, Resettable)
    system.reset()
    system.close()

    assert events == ["cleared", "closed"]


# --- building the systems ---


@pytest.mark.parametrize(("factory", "crawls"), [(create_plain_system, False), (create_crawl_system, True)])
def test_the_systems_are_built_without_links_and_with_a_patient_retrier(
    monkeypatch: pytest.MonkeyPatch, factory: Any, crawls: bool
) -> None:
    built: list[dict[str, Any]] = []
    client = FakeClient()
    runtime = SimpleNamespace(
        ingestor=FakeIngestor(), retriever=FakeRetriever([]), answer_model=GroqChatModel(client), close=lambda: None
    )

    def fake_build_runtime(settings: object, usage: UsageCounter, **options: Any) -> SimpleNamespace:
        built.append({"settings": settings, "usage": usage, **options})
        return runtime

    monkeypatch.setattr(baselines, "build_runtime", fake_build_runtime)
    settings = SimpleNamespace(eval_crawl_max_pages=7, eval_crawl_max_depth=3)

    system = factory(cast("Any", settings), UsageCounter())

    assert isinstance(system, PlainSystem)
    (call,) = built
    assert call["follow_links"] is False
    assert call["check_followed"] is False
    assert call["retrier"] is not None
    assert system._crawl == (CrawlLimits(max_pages=7, max_depth=3) if crawls else None)


def test_the_baselines_are_registered_for_the_evaluation() -> None:
    assert SYSTEMS["plain"] is create_plain_system
    assert SYSTEMS["crawl"] is create_crawl_system
    assert sorted(SYSTEMS) == ["agent", "agent-single", "crawl", "plain"]
