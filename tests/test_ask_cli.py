"""Tests of the ``agent-ask`` command, with fakes instead of the agent and the services."""

from types import SimpleNamespace
from typing import Any

import pytest

from chat_with_any_website_agent import ask_cli
from chat_with_any_website_agent.ask_cli import (
    EXIT_BAD_SETTINGS,
    EXIT_BAD_SOURCE,
    EXIT_OK,
    EXIT_SERVICE,
    format_answer,
    main,
)
from chat_with_any_website_agent.citations import Citation
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import (
    BudgetExceededError,
    ExtractionError,
    FetchError,
    InvalidURLError,
    ServiceBusyError,
)
from chat_with_any_website_agent.graph import AgentAnswer
from chat_with_any_website_agent.ingest import IngestResult
from chat_with_any_website_agent.usage import COHERE, GROQ, UsageCounter

URL = "https://docs.example/control"
NORMALIZED = f"{URL}/"
GROQ_KEY = "groq-secret-key"
CITATION = Citation(number=1, url=URL, title="Control Flow", heading="pass Statements", text="The pass statement.")
ANSWERED = AgentAnswer(
    answer="The pass statement does nothing [1].",
    abstained=False,
    citations=(CITATION,),
    steps=("Read the question.", "Wrote an answer."),
    llm_calls=2,
)
ABSTAINED = AgentAnswer(
    answer="The source does not cover this question, so there is no answer to give from it.",
    abstained=True,
    reason="The excerpts do not answer the question.",
    steps=("Read the question.", "Decided not to answer: The excerpts do not answer the question."),
)


class FakeRuntime:
    """An agent and an ingestor that record how they were used, and that can fail."""

    def __init__(self, answer: AgentAnswer = ANSWERED, error: BaseException | None = None) -> None:
        self.answer = answer
        self.error = error
        self.closed = False
        self.ingested: list[tuple[str, bool]] = []
        self.asked: list[tuple[str, list[str]]] = []
        self.ingestor = SimpleNamespace(ingest=self._ingest)
        self.agent = SimpleNamespace(ask=self._ask)

    def _ingest(self, url: str, *, refresh: bool = False) -> IngestResult:
        self.ingested.append((url, refresh))
        if self.error is not None:
            raise self.error
        return IngestResult(url=f"{url}/", reused=False, title="Control Flow", chunk_count=4)  # normalized

    def _ask(self, question: str, urls: list[str]) -> AgentAnswer:
        self.asked.append((question, urls))
        return self.answer

    def close(self) -> None:
        self.closed = True


def _wire(monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime | None = None) -> tuple[FakeRuntime, list[Any]]:
    for name, value in {
        "COHERE_API_KEY": "cohere-secret-key",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "weaviate-secret-key",
        "GROQ_API_KEY": GROQ_KEY,
    }.items():
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    runtime = runtime or FakeRuntime()
    built: list[Any] = []

    def build_runtime(settings: Settings, usage: UsageCounter, **options: Any) -> FakeRuntime:
        built.append(options)
        return runtime

    monkeypatch.setattr(ask_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(ask_cli, "build_runtime", build_runtime)
    return runtime, built


# --- what is printed ---


def test_the_answer_comes_with_its_sources_and_the_steps_of_the_agent() -> None:
    usage = UsageCounter()
    usage.record_call(COHERE)
    usage.record_call(GROQ)
    usage.record_call(GROQ)
    usage.record_tokens(GROQ, 1234)
    ingested = IngestResult(url=NORMALIZED, reused=False, title="Control Flow", chunk_count=4)

    text = format_answer(ingested, ANSWERED, usage, 3.456)

    assert text == "\n".join(
        [
            "Loaded the source (4 chunk(s) stored).",
            "",
            "The pass statement does nothing [1].",
            "",
            "Sources cited:",
            f"  [1] Control Flow > pass Statements ({URL})",
            "",
            "Steps:",
            "  1. Read the question.",
            "  2. Wrote an answer.",
            "",
            "Cohere calls: 1. Groq calls: 2. Groq tokens: 1234. Time: 3.5 s.",
        ]
    )


def test_a_stored_source_is_said_to_be_reused() -> None:
    text = format_answer(IngestResult(url=NORMALIZED, reused=True), ANSWERED, UsageCounter(), 1.0)

    assert text.startswith("Reused the stored copy of the source.")


def test_an_abstention_gives_the_reason_and_no_sources() -> None:
    text = format_answer(IngestResult(url=NORMALIZED, reused=True), ABSTAINED, UsageCounter(), 1.0)

    assert "Reason: The excerpts do not answer the question." in text
    assert "Sources cited" not in text
    assert "rejected" not in text


def test_an_answer_that_the_check_rejected_is_shown_with_the_reason() -> None:
    rejected = AgentAnswer(
        answer=ABSTAINED.answer,
        abstained=True,
        reason="The answer cites no excerpt.",
        draft="The else clause is not run after a break.",
    )

    text = format_answer(IngestResult(url=NORMALIZED, reused=True), rejected, UsageCounter(), 1.0)

    assert "Reason: The answer cites no excerpt." in text
    assert "The answer that the check rejected:\n  The else clause is not run after a break." in text


def test_the_steps_can_be_left_out() -> None:
    text = format_answer(IngestResult(url=NORMALIZED, reused=True), ANSWERED, UsageCounter(), 1.0, show_steps=False)

    assert "Steps:" not in text


def test_a_source_without_a_section_is_listed_by_its_title() -> None:
    answer = AgentAnswer(
        answer="Yes [1].",
        abstained=False,
        citations=(Citation(number=1, url=URL, title="Control Flow", heading="", text="t"),),
    )

    text = format_answer(IngestResult(url=NORMALIZED, reused=True), answer, UsageCounter(), 1.0)

    assert f"  [1] Control Flow ({URL})" in text


# --- the command ---


def test_the_source_is_loaded_and_the_question_asked_about_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime, built = _wire(monkeypatch)

    code = main([URL, "What does pass do?"])

    assert code == EXIT_OK
    assert runtime.ingested == [(URL, False)]
    assert runtime.asked == [("What does pass do?", [NORMALIZED])]  # the normalized address the source is stored under
    assert built == [{"tenant": None}]
    assert runtime.closed
    out = capsys.readouterr().out
    assert "The pass statement does nothing [1]." in out
    assert "Steps:" in out


def test_further_sources_are_loaded_and_searched_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime, _ = _wire(monkeypatch)
    other, third = "https://docs.example/data", "https://docs.example/errors"

    code = main([URL, "Which one?", "--also", other, "--also", third, "--refresh"])

    assert code == EXIT_OK
    assert runtime.ingested == [(URL, True), (other, True), (third, True)]
    assert runtime.asked == [("Which one?", [NORMALIZED, f"{other}/", f"{third}/"])]
    out = capsys.readouterr().out
    assert "Loaded the source (4 chunk(s) stored)." in out
    assert f"Loaded {other}/ (4 chunk(s) stored)." in out
    assert f"Loaded {third}/ (4 chunk(s) stored)." in out


def test_the_stored_copies_of_further_sources_are_said_to_be_reused() -> None:
    first = IngestResult(url=NORMALIZED, reused=True)
    extra = IngestResult(url="https://docs.example/data/", reused=True)

    text = format_answer(first, ANSWERED, UsageCounter(), 1.0, extras=[extra])

    assert text.startswith(
        "Reused the stored copy of the source.\nReused the stored copy of https://docs.example/data/.\n\n"
    )


def test_an_abstention_is_not_an_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _wire(monkeypatch, FakeRuntime(ABSTAINED))

    code = main([URL, "Q?"])

    assert code == EXIT_OK
    assert "Reason: The excerpts do not answer the question." in capsys.readouterr().out


def test_the_options_reach_the_runtime_and_the_ingestor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime, built = _wire(monkeypatch)

    main([URL, "Q?", "--tenant", "demo", "--refresh", "--no-steps"])

    assert built == [{"tenant": "demo"}]
    assert runtime.ingested == [(URL, True)]
    assert "Steps:" not in capsys.readouterr().out


def test_settings_that_are_not_valid_are_named_without_their_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def invalid() -> Settings:
        return Settings(_env_file=None)  # type: ignore[call-arg]

    monkeypatch.setattr(ask_cli, "get_settings", invalid)
    monkeypatch.setattr(ask_cli, "build_runtime", lambda *args, **options: pytest.fail("no service may be used"))

    code = main([URL, "Q?"])

    assert code == EXIT_BAD_SETTINGS
    assert "The settings are not valid (COHERE_API_KEY: Field required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [InvalidURLError("That address is not allowed."), FetchError("Download failed."), ExtractionError("No text.")],
)
def test_a_source_that_cannot_be_used_is_reported_and_the_connections_are_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: BaseException
) -> None:
    runtime, _ = _wire(monkeypatch, FakeRuntime(error=error))

    code = main([URL, "Q?"])

    assert code == EXIT_BAD_SOURCE
    assert str(error) in capsys.readouterr().err
    assert runtime.closed
    assert runtime.asked == []


@pytest.mark.parametrize(
    "error", [ServiceBusyError("Groq is busy.", retry_after=30.0), BudgetExceededError("Limit reached.", "groq", 10)]
)
def test_a_busy_service_or_a_reached_limit_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: BaseException
) -> None:
    runtime, _ = _wire(monkeypatch, FakeRuntime(error=error))

    code = main([URL, "Q?"])

    assert code == EXIT_SERVICE
    assert str(error) in capsys.readouterr().err
    assert runtime.closed


def test_an_unexpected_failure_is_described_without_any_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime, _ = _wire(monkeypatch, FakeRuntime(error=RuntimeError(f"401 for the key {GROQ_KEY}")))

    code = main([URL, "Q?"])

    err = capsys.readouterr().err
    assert code == EXIT_SERVICE
    assert "A service failed: RuntimeError: 401 for the key ***" in err
    assert GROQ_KEY not in err
    assert runtime.closed


def test_a_service_that_cannot_be_reached_at_the_start_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wire(monkeypatch)

    def unreachable(*args: Any, **options: Any) -> None:
        raise ConnectionError("Weaviate cannot be reached")

    monkeypatch.setattr(ask_cli, "build_runtime", unreachable)

    code = main([URL, "Q?"])

    assert code == EXIT_SERVICE
    assert "A service failed: ConnectionError: Weaviate cannot be reached" in capsys.readouterr().err
