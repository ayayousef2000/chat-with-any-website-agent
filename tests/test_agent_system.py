"""Tests of the agent as a system for the evaluation, with fakes instead of the agent and the services."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from chat_with_any_website_agent import agent_system
from chat_with_any_website_agent.agent_system import AgentSystem, create_agent_system
from chat_with_any_website_agent.cohere_support import COHERE_RETRYABLE
from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError, ServiceBusyError
from chat_with_any_website_agent.evaluation import Case, Outcome
from chat_with_any_website_agent.graph import AgentAnswer
from chat_with_any_website_agent.groq_client import GROQ_RETRYABLE
from chat_with_any_website_agent.ingest import IngestResult
from chat_with_any_website_agent.retrying import Retrier
from chat_with_any_website_agent.systems import SYSTEMS
from chat_with_any_website_agent.usage import UsageCounter

if TYPE_CHECKING:
    from chat_with_any_website_agent.wiring import AgentRuntime

URL = "https://docs.example/control"
OTHER = "https://docs.example/data"


class FakeIngestor:
    """Stores nothing; it normalizes the address by adding a slash and can fail for some of them."""

    def __init__(self, failures: dict[str, BaseException] | None = None) -> None:
        self.failures = failures or {}
        self.ingested: list[str] = []
        self.checks: list[bool] = []  # whether the website was to be asked about a change, for each source

    def ingest(self, url: str, *, check: bool = True) -> IngestResult:
        self.ingested.append(url)
        self.checks.append(check)
        if url in self.failures:
            raise self.failures[url]
        return IngestResult(url=f"{url}/", reused=True)


ANSWERED = AgentAnswer(
    answer="The answer [1].", abstained=False, excerpts=("An excerpt.",), sources=(f"{URL}/",), llm_calls=2
)


class FakeAgent:
    """Gives the answer it was given to every question and records what it was asked."""

    def __init__(self, answer: AgentAnswer = ANSWERED) -> None:
        self.answer = answer
        self.asked: list[tuple[str, list[str]]] = []

    def ask(self, question: str, urls: list[str]) -> AgentAnswer:
        self.asked.append((question, urls))
        return self.answer


def _system(
    ingestor: FakeIngestor | None = None, answer: AgentAnswer = ANSWERED
) -> tuple[AgentSystem, FakeIngestor, FakeAgent, list[str]]:
    fake_ingestor = ingestor or FakeIngestor()
    agent = FakeAgent(answer)
    closed: list[str] = []
    runtime = SimpleNamespace(ingestor=fake_ingestor, agent=agent, close=lambda: closed.append("closed"))
    return AgentSystem(cast("AgentRuntime", runtime)), fake_ingestor, agent, closed


def test_a_question_is_answered_about_its_source_and_the_sources_it_names() -> None:
    system, ingestor, agent, _ = _system()
    case = Case(url=URL, question="Which one?", sources=(OTHER,))

    outcome = system.ask(case)

    assert ingestor.ingested == [URL, OTHER]
    assert agent.asked == [("Which one?", [f"{URL}/", f"{OTHER}/"])]  # the normalized addresses, in order
    assert outcome == Outcome(answer="The answer [1].", excerpts=("An excerpt.",), sources=(f"{URL}/",))


def test_a_question_does_not_ask_the_website_again_but_preparing_the_source_does() -> None:
    system, ingestor, _, _ = _system()

    system.prepare(URL)
    system.ask(Case(url=URL, question="Q?", sources=(OTHER,)))

    # The source was checked when the run prepared it. The questions use the copy that the run began with, so that
    # the answers of one run do not depend on when each question was asked.
    assert ingestor.checks == [True, False, False]


def test_a_source_that_cannot_be_loaded_is_a_failed_answer_and_not_an_error() -> None:
    failure = FetchError("The page could not be downloaded.")
    system, _, agent, _ = _system(FakeIngestor({URL: failure}))

    outcome = system.ask(Case(url=URL, question="Q?"))

    assert outcome == Outcome(
        answer="The source could not be loaded: The page could not be downloaded.",
        reason="The page could not be downloaded.",
    )
    assert agent.asked == []


def test_why_the_agent_did_not_answer_and_how_it_got_there_are_passed_on() -> None:
    refused = AgentAnswer(
        answer="The source does not cover this question, so there is no answer to give from it.",
        abstained=True,
        reason="The answer cites no excerpt.",
        draft="A rejected answer.",
        excerpts=("An excerpt.",),
        steps=("Read the question.", "Decided not to answer: The answer cites no excerpt."),
    )
    system, _, _, _ = _system(answer=refused)

    outcome = system.ask(Case(url=URL, question="Q?"))

    assert outcome.reason == "The answer cites no excerpt."
    assert outcome.draft == "A rejected answer."
    assert outcome.steps == ("Read the question.", "Decided not to answer: The answer cites no excerpt.")
    assert outcome.excerpts == ("An excerpt.",)


@pytest.mark.parametrize("error", [InvalidURLError("bad address"), ExtractionError("no text")])
def test_every_way_a_source_can_be_unreadable_is_a_failed_answer(error: BaseException) -> None:
    system, _, _, _ = _system(FakeIngestor({URL: error}))

    assert system.ask(Case(url=URL, question="Q?")).answer.startswith("The source could not be loaded")


def test_a_second_source_that_cannot_be_loaded_fails_the_question() -> None:
    system, _, agent, _ = _system(FakeIngestor({OTHER: FetchError("gone")}))

    outcome = system.ask(Case(url=URL, question="Q?", sources=(OTHER,)))

    assert outcome.answer == "The source could not be loaded: gone"
    assert agent.asked == []


def test_a_busy_service_stops_the_run_instead_of_being_recorded() -> None:
    busy = ServiceBusyError("Cohere is busy.", retry_after=30.0)
    system, _, _, _ = _system(FakeIngestor({URL: busy}))

    with pytest.raises(ServiceBusyError):
        system.ask(Case(url=URL, question="Q?"))
    with pytest.raises(ServiceBusyError):
        system.prepare(URL)


def test_preparing_a_source_loads_it() -> None:
    system, ingestor, _, _ = _system()

    system.prepare(URL)

    assert ingestor.ingested == [URL]


def test_a_source_that_cannot_be_prepared_does_not_stop_the_run() -> None:
    system, _, _, _ = _system(FakeIngestor({URL: FetchError("gone")}))

    system.prepare(URL)  # the question about it will report the problem


def test_closing_the_system_closes_the_runtime() -> None:
    system, _, _, closed = _system()

    system.close()

    assert closed == ["closed"]


def test_the_agent_is_registered_for_the_evaluation() -> None:
    assert SYSTEMS["agent"] is create_agent_system


def test_the_system_for_a_run_is_built_with_a_patient_retrier(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []
    runtime = SimpleNamespace(ingestor=FakeIngestor(), agent=FakeAgent(), close=lambda: None)

    def fake_build_runtime(settings: object, usage: UsageCounter, **options: Any) -> SimpleNamespace:
        built.append({"settings": settings, "usage": usage, **options})
        return runtime

    monkeypatch.setattr(agent_system, "build_runtime", fake_build_runtime)
    usage, settings = UsageCounter(), object()

    system = create_agent_system(cast("Any", settings), usage)

    assert isinstance(system, AgentSystem)
    (call,) = built
    assert call["settings"] is settings
    assert call["usage"] is usage
    assert isinstance(call["retrier"], Retrier)
    assert call["retrier"].retryable == GROQ_RETRYABLE
    assert call["retrier"].policy.budget > 45.0
    assert isinstance(call["cohere_retrier"], Retrier)
    assert call["cohere_retrier"].retryable == COHERE_RETRYABLE
    assert call["cohere_retrier"].policy.budget > 45.0
    assert call["check_followed"] is False  # the pages of one run are not asked about again while it goes on


def test_the_agent_without_links_is_built_without_the_ability_to_follow_them(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []
    runtime = SimpleNamespace(ingestor=FakeIngestor(), agent=FakeAgent(), close=lambda: None)

    def fake_build_runtime(settings: object, usage: UsageCounter, **options: Any) -> SimpleNamespace:
        built.append(options)
        return runtime

    monkeypatch.setattr(agent_system, "build_runtime", fake_build_runtime)

    system = agent_system.create_single_page_agent_system(cast("Any", object()), UsageCounter())

    assert isinstance(system, AgentSystem)
    (options,) = built
    assert options["follow_links"] is False
    assert options["check_followed"] is False
    assert isinstance(options["retrier"], Retrier)
    assert isinstance(options["cohere_retrier"], Retrier)
    assert options["cohere_retrier"].policy.budget > 45.0
    assert SYSTEMS["agent-single"] is agent_system.create_single_page_agent_system


def test_the_agent_system_can_empty_the_tenant() -> None:
    cleared: list[str] = []
    runtime = SimpleNamespace(
        ingestor=FakeIngestor(), agent=FakeAgent(), store=SimpleNamespace(clear=lambda: cleared.append("x"))
    )

    AgentSystem(cast("AgentRuntime", runtime)).reset()

    assert cleared == ["x"]
