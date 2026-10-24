"""Tests of the optional local trace of the questions, with scripted models and fakes."""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.graph import AgentAnswer, AgentLimits, SourceAgent
from chat_with_any_website_agent.retriever import SourceRetriever
from chat_with_any_website_agent.trace import append_trace, trace_record
from tests.test_graph import (
    CITED_ANSWER,
    ENOUGH,
    FIRST_SEARCH,
    LIMITS,
    NOT_ENOUGH,
    QUESTION,
    URL,
    FakeEmbedder,
    FakeReranker,
    FakeStore,
    ScriptedClient,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

ANSWER = AgentAnswer(
    answer="Pass does nothing [1].",
    abstained=False,
    steps=("Read the question.", "Wrote an answer."),
    followed=("https://docs.example/more",),
    sources=(URL,),
    llm_calls=2,
)


def _agent(trace_file: str, grader: list[str], answerer: list[str], limits: AgentLimits = LIMITS) -> SourceAgent:
    retriever = SourceRetriever(FakeStore([FIRST_SEARCH]), FakeEmbedder(), FakeReranker())
    return SourceAgent(
        retriever,
        GroqChatModel(ScriptedClient(*answerer)),
        GroqChatModel(ScriptedClient(*grader)),
        limits,
        trace_file=trace_file,
    )


def test_a_record_names_the_question_and_what_the_agent_did_but_not_the_text_of_the_answer() -> None:
    now = datetime(2026, 10, 24, 9, 30, 5, tzinfo=UTC)

    assert trace_record(QUESTION, ANSWER, now=now) == {
        "time": "2026-10-24T09:30:05+00:00",
        "question": QUESTION,
        "abstained": False,
        "reason": None,
        "steps": ["Read the question.", "Wrote an answer."],
        "followed": ["https://docs.example/more"],
        "sources": [URL],
        "llm_calls": 2,
    }


def test_each_question_adds_one_line_and_the_lines_are_kept(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"

    append_trace(str(path), "Première question ?", ANSWER)
    append_trace(str(path), "Second question?", ANSWER)

    raw = path.read_bytes()
    assert b"\r" not in raw  # the line ends are the same on every system
    lines = raw.decode("utf-8").splitlines()
    assert [json.loads(line)["question"] for line in lines] == ["Première question ?", "Second question?"]
    assert "Première" in lines[0]  # written as text, not as escapes


def test_a_blank_file_name_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    append_trace("  ", QUESTION, ANSWER)

    assert list(tmp_path.iterdir()) == []


def test_a_file_that_cannot_be_written_is_logged_and_does_not_fail_the_question(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    append_trace(str(tmp_path / "missing" / "trace.jsonl"), QUESTION, ANSWER)

    assert "The trace file could not be written: FileNotFoundError" in caplog.text
    assert QUESTION not in caplog.text  # a log line never repeats the question


def test_an_answered_question_is_traced_with_its_steps(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    agent = _agent(str(path), [ENOUGH], [CITED_ANSWER])

    result = agent.ask(f"  {QUESTION}  ", [URL])

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["question"] == QUESTION  # as the agent read it, tidied
    assert record["abstained"] is False
    assert record["steps"] == list(result.steps)
    assert record["sources"] == [URL]
    assert record["llm_calls"] == 2
    assert CITED_ANSWER not in path.read_text(encoding="utf-8")


def test_a_refused_question_is_traced_with_its_reason(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    agent = _agent(str(path), [NOT_ENOUGH], [], AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=0))

    result = agent.ask(QUESTION, [URL])

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["abstained"] is True
    assert record["reason"] == result.reason
    assert record["reason"]


def test_nothing_is_traced_unless_a_file_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    _agent("", [ENOUGH], [CITED_ANSWER]).ask(QUESTION, [URL])

    assert list(tmp_path.iterdir()) == []
