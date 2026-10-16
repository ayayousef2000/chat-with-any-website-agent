"""Tests of the command line of the evaluation. Nothing here uses a real service."""

import json
from typing import TYPE_CHECKING, Any

from chat_with_any_website_agent import evaluation_cli
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.evaluation import Outcome, Result, save_results
from chat_with_any_website_agent.evaluation_cli import EXIT_BAD_INPUT, EXIT_OK, EXIT_STOPPED, EXIT_UNKNOWN_SYSTEM, main
from chat_with_any_website_agent.groq_client import ChatReply
from chat_with_any_website_agent.usage import GROQ

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from chat_with_any_website_agent.evaluation import Case
    from chat_with_any_website_agent.usage import UsageCounter

PAGES = [
    {
        "url": "https://a.example/",
        "questions": [
            {"question": "q1", "reference": "r1", "category": "multi_page", "language": "ar"},
            {"question": "q2", "answerable": False},
        ],
    }
]


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _result(question: str, **kwargs: Any) -> Result:
    values: dict[str, Any] = {
        "url": "https://a.example/",
        "question": question,
        "category": "single_page",
        "language": "en",
        "answerable": True,
        "retrieval_hit": True,
        "faithful": True,
        "correct": True,
        "reason": "",
        "answer": "an answer",
        "seconds": 1.0,
        "cohere_calls": 1,
        "groq_calls": 1,
        "groq_tokens": 10,
    }
    values.update(kwargs)
    return Result(**values)


# --- validate and summary: files only ---


def test_validate_counts_the_questions_by_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "questions.json", PAGES)

    assert main(["validate", str(path)]) == EXIT_OK

    output = capsys.readouterr().out
    assert "2 questions (1 answerable, 1 not)" in output
    assert "categories: multi_page 1, single_page 1" in output
    assert "languages: ar 1, en 1" in output


def test_validate_reports_what_is_wrong_and_where(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "questions.json", [{"url": "u", "questions": [{"question": "q"}]}])

    assert main(["validate", str(path)]) == EXIT_BAD_INPUT

    assert "page 1, question 1" in capsys.readouterr().err


def test_summary_combines_several_results_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    save_results(first, [_result("q1")])
    save_results(second, [_result("q2", correct=False)])

    assert main(["summary", str(first), str(second)]) == EXIT_OK

    row = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("all"))
    assert row.split()[:5] == ["all", "2", "100%", "100%", "50%"]


def test_summary_of_a_file_that_is_not_results_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "other.json", {"a": 1})

    assert main(["summary", str(path)]) == EXIT_BAD_INPUT

    assert "does not hold evaluation results" in capsys.readouterr().err


# --- run ---


def test_an_unknown_system_is_refused_before_anything_is_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluation_cli, "SYSTEMS", {})
    path = _write(tmp_path / "questions.json", PAGES)

    code = main(["run", str(path), "--system", "nope", "--out", str(tmp_path / "out.json")])

    assert code == EXIT_UNKNOWN_SYSTEM
    assert "Unknown system 'nope'. Available: none yet." in capsys.readouterr().err


class FakeSystem:
    def __init__(self, usage: UsageCounter) -> None:
        self.usage = usage
        self.closed = False

    def prepare(self, url: str) -> None:
        pass

    def ask(self, case: Case) -> Outcome:
        self.usage.record_call(GROQ)
        return Outcome(answer=f"answer to {case.question}", excerpts=("an excerpt",))

    def close(self) -> None:
        self.closed = True


class FakeJudge:
    def complete(self, messages: Any, **options: Any) -> ChatReply:
        verdict = '{"faithful": true, "correct": true, "reason": "fine"}'
        return ChatReply(text=verdict, prompt_tokens=1, completion_tokens=1, key_number=1)


class FakeGroqClient:
    models: list[str] = []  # noqa: RUF012 - a test double that records how it was asked

    @classmethod
    def from_settings(cls, settings: Settings, *, model: str, usage: UsageCounter, retrier: Any) -> FakeJudge:
        cls.models.append(model)
        return FakeJudge()


def _wire(monkeypatch: pytest.MonkeyPatch, systems: dict[str, Any], **env: str) -> list[FakeSystem]:
    built: list[FakeSystem] = []

    def factory(settings: Settings, usage: UsageCounter) -> FakeSystem:
        system = FakeSystem(usage)
        built.append(system)
        return system

    for name, value in {
        "COHERE_API_KEY": "c",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "w",
        "GROQ_API_KEY": "g",
        **env,
    }.items():
        monkeypatch.setenv(name, value)
    systems["fake"] = factory
    FakeGroqClient.models = []
    monkeypatch.setattr(evaluation_cli, "SYSTEMS", systems)
    monkeypatch.setattr(evaluation_cli, "get_settings", lambda: Settings(_env_file=None))  # type: ignore[call-arg]
    monkeypatch.setattr(evaluation_cli, "GroqClient", FakeGroqClient)
    return built


def test_a_run_saves_the_results_prints_a_summary_and_closes_the_system(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _wire(monkeypatch, {})
    out = tmp_path / "out.json"

    code = main(["run", str(_write(tmp_path / "questions.json", PAGES)), "--system", "fake", "--out", str(out)])

    assert code == EXIT_OK
    assert len(json.loads(out.read_text())) == 2
    output = capsys.readouterr().out
    assert "This run: 2 question(s)." in output
    assert "multi_page" in output
    assert built[0].closed is True
    assert FakeGroqClient.models == ["openai/gpt-oss-20b"]  # the judge model of the settings


def test_a_run_that_reaches_its_limit_says_so_and_can_be_continued(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire(monkeypatch, {}, EVAL_MAX_GROQ_CALLS="1")
    out = tmp_path / "out.json"
    dataset = str(_write(tmp_path / "questions.json", PAGES))

    code = main(["run", dataset, "--system", "fake", "--out", str(out)])

    assert code == EXIT_STOPPED
    assert len(json.loads(out.read_text())) == 1
    assert "stopped early" in capsys.readouterr().err

    _wire(monkeypatch, {})
    assert main(["run", dataset, "--system", "fake", "--out", str(out), "--resume"]) == EXIT_OK
    assert len(json.loads(out.read_text())) == 2


def test_a_run_can_be_limited_to_a_few_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, {})
    out = tmp_path / "out.json"

    dataset = str(_write(tmp_path / "questions.json", PAGES))
    main(["run", dataset, "--system", "fake", "--out", str(out), "--limit", "1"])

    assert len(json.loads(out.read_text())) == 1
