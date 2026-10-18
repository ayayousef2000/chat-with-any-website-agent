"""Tests of the evaluation harness with a fake system and a fake judge: no service is used."""

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import BudgetExceededError, DatasetError, ServiceBusyError
from chat_with_any_website_agent.evaluation import (
    JUDGE_PROMPT,
    NOT_ANSWERABLE,
    Case,
    Outcome,
    Result,
    Turn,
    evaluation_usage,
    judge_answer,
    load_cases,
    load_results,
    parse_verdict,
    run_and_save,
    run_evaluation,
    save_results,
)
from chat_with_any_website_agent.groq_client import ChatReply
from chat_with_any_website_agent.usage import COHERE, GROQ, UsageCounter

if TYPE_CHECKING:
    from pathlib import Path

GOOD_VERDICT = '{"faithful": true, "correct": true, "reason": "fine"}'

PAGES = [
    {
        "url": "https://a.example/",
        "questions": [
            {
                "question": "q1",
                "reference": "r1",
                "must_contain": ["Alpha"],
                "category": "multi_page",
                "language": "ar",
                "sources": ["https://b.example/"],
                "history": [{"question": "earlier", "answer": "reply"}],
            },
            {"question": "q2", "answerable": False},
        ],
    },
    {"url": "https://b.example/", "questions": [{"question": "q3", "reference": "r3"}]},
]


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- the question set ---


def test_a_question_set_is_read_with_its_defaults(tmp_path: Path) -> None:
    cases = load_cases(_write(tmp_path / "questions.json", PAGES))

    assert [case.question for case in cases] == ["q1", "q2", "q3"]
    first, second, third = cases
    assert first == Case(
        url="https://a.example/",
        question="q1",
        reference="r1",
        must_contain=("Alpha",),
        history=(Turn("earlier", "reply"),),
        category="multi_page",
        language="ar",
        sources=("https://b.example/",),
    )
    assert (second.answerable, second.reference, second.category, second.language) == (False, None, "single_page", "en")
    assert third.url == "https://b.example/"
    assert first.key == "https://a.example/\nq1"


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"url": "x"}, "must hold a list of pages"),
        (["page"], "a page must be an object"),
        ([{"questions": [{"question": "q", "reference": "r"}]}], "url must be a non-empty text"),
        ([{"url": "u", "questions": []}], "non-empty list of questions"),
        ([{"url": "u", "questions": ["q"]}], "a question must be an object"),
        ([{"url": "u", "questions": [{"reference": "r"}]}], "question must be a non-empty text"),
        ([{"url": "u", "questions": [{"question": "q"}]}], "an answerable question needs one"),
        ([{"url": "u", "questions": [{"question": "q", "reference": "r", "answerable": "no"}]}], "true or false"),
        (
            [{"url": "u", "questions": [{"question": "q", "reference": "r", "history": "x"}]}],
            "history must be a list",
        ),
        (
            [{"url": "u", "questions": [{"question": "q", "reference": "r", "history": ["x"]}]}],
            "must be an object",
        ),
        (
            [{"url": "u", "questions": [{"question": "q", "reference": "r", "history": [{"question": "a"}]}]}],
            "history answer",
        ),
        ([{"url": "u", "questions": [{"question": "q", "reference": "r", "must_contain": "x"}]}], "must be a list"),
        (
            [{"url": "u", "questions": [{"question": "q", "reference": "r"}, {"question": "q", "reference": "r"}]}],
            "asked twice",
        ),
    ],
)
def test_a_question_set_that_does_not_follow_the_format_is_refused_with_where_and_why(
    tmp_path: Path, data: Any, fragment: str
) -> None:
    with pytest.raises(DatasetError, match=fragment):
        load_cases(_write(tmp_path / "questions.json", data))


def test_the_message_says_which_page_and_question_is_wrong(tmp_path: Path) -> None:
    data = [{"url": "u", "questions": [{"question": "ok", "reference": "r"}, {"question": "bad"}]}]

    with pytest.raises(DatasetError, match=r"page 1, question 2"):
        load_cases(_write(tmp_path / "questions.json", data))


def test_a_file_that_is_not_json_or_does_not_exist_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(DatasetError, match="cannot be read as JSON"):
        load_cases(broken)
    with pytest.raises(DatasetError, match="cannot be read as JSON"):
        load_cases(tmp_path / "missing.json")


# --- the judge ---


class FakeJudge:
    """A stand-in for the judge's model client."""

    def __init__(self, text: str = GOOD_VERDICT, usage: UsageCounter | None = None) -> None:
        self.text = text
        self.usage = usage
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    def complete(self, messages: Any, **options: Any) -> ChatReply:
        self.calls.append((list(messages), options))
        if self.usage is not None:
            self.usage.record_call(GROQ)  # the judge is counted, but not as part of the cost of the system
        return ChatReply(text=self.text, prompt_tokens=1, completion_tokens=1, key_number=1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (GOOD_VERDICT, (True, True, "fine")),
        ('{"faithful": null, "correct": false, "reason": "no excerpts"}', (None, False, "no excerpts")),
        ('{"correct": true}', (None, True, "")),
    ],
)
def test_the_judges_reply_is_read(text: str, expected: tuple[bool | None, bool, str]) -> None:
    verdict = parse_verdict(text)

    assert verdict is not None
    assert (verdict.faithful, verdict.correct, verdict.reason) == expected


@pytest.mark.parametrize(
    "text",
    ["not json", "[]", '"yes"', "{}", '{"correct": "yes"}', '{"correct": true, "faithful": "maybe"}'],
)
def test_a_reply_that_cannot_be_read_is_not_a_verdict(text: str) -> None:
    assert parse_verdict(text) is None


def test_the_judge_is_given_the_question_the_reference_the_excerpts_and_the_answer() -> None:
    judge = FakeJudge()
    case = Case(url="https://a.example/", question="q1", reference="the reference")
    outcome = Outcome(answer="the answer", excerpts=("first excerpt", "second excerpt"))

    verdict = judge_answer(judge, case, outcome)

    assert verdict is not None
    messages, options = judge.calls[0]
    assert messages[0] == {"role": "system", "content": JUDGE_PROMPT}
    user = messages[1]["content"]
    assert "Question: q1" in user
    assert "Reference answer: the reference" in user
    assert "[1] first excerpt\n\n[2] second excerpt" in user
    assert "System answer: the answer" in user
    assert options == {"response_format": {"type": "json_object"}, "temperature": 0}


def test_an_unanswerable_question_has_no_reference_and_a_missing_excerpt_list_is_marked() -> None:
    judge = FakeJudge()

    judge_answer(judge, Case(url="u", question="q", answerable=False), Outcome(answer="a"))

    user = judge.calls[0][0][1]["content"]
    assert f"Reference answer: {NOT_ANSWERABLE}" in user
    assert "Retrieved excerpts:\n(none)" in user


# --- running ---


class FakeSystem:
    """A stand-in for the system under test that answers from a text with its question in it."""

    def __init__(self, usage: UsageCounter) -> None:
        self.usage = usage
        self.prepared: list[str] = []
        self.asked: list[str] = []
        self.failures: dict[str, BaseException] = {}
        self.closed = False
        self.watch: Path | None = None
        self.saved_when_asked: list[int] = []

    def prepare(self, url: str) -> None:
        self.prepared.append(url)

    def ask(self, case: Case) -> Outcome:
        self.asked.append(case.question)
        if self.watch is not None:
            self.saved_when_asked.append(len(json.loads(self.watch.read_text())) if self.watch.exists() else 0)
        if case.question in self.failures:
            raise self.failures[case.question]
        self.usage.record_call(COHERE)
        self.usage.record_call(GROQ)
        self.usage.record_tokens(GROQ, 100)
        return Outcome(answer=f"answer to {case.question}", excerpts=(f"a text with alpha for {case.question}",))

    def close(self) -> None:
        self.closed = True


def _case(question: str, url: str = "https://a.example/", **kwargs: Any) -> Case:
    kwargs.setdefault("reference", "the reference")
    return Case(url=url, question=question, **kwargs)


def test_every_question_is_scored_and_what_it_cost_is_measured_without_the_judge() -> None:
    usage = UsageCounter()
    system = FakeSystem(usage)
    judge = FakeJudge(usage=usage)
    cases = [
        _case("q1", must_contain=("ALPHA",)),
        _case("q2", must_contain=("missing phrase",)),
        _case("q3", url="https://b.example/", answerable=False, reference=None),
    ]
    seen: list[Result] = []

    report = run_evaluation(system, cases, judge, usage, on_result=seen.append)

    assert report.stopped_because is None
    assert report.evaluated == 3
    assert report.results == seen
    assert system.prepared == ["https://a.example/", "https://b.example/"]  # each source once
    first, second, third = report.results
    assert (first.retrieval_hit, second.retrieval_hit, third.retrieval_hit) == (True, False, None)
    assert (first.faithful, first.correct, first.reason) == (True, True, "fine")
    assert first.answer == "answer to q1"
    assert (first.cohere_calls, first.groq_calls, first.groq_tokens) == (1, 1, 100)  # the judge's call is not in it
    assert usage.calls == {COHERE: 3, GROQ: 6}  # but the run's counter has all of them
    assert first.seconds >= 0


def test_an_answer_with_no_excerpts_has_no_retrieval_score() -> None:
    usage = UsageCounter()

    class Remote(FakeSystem):
        def ask(self, case: Case) -> Outcome:
            return Outcome(answer="an answer")

    report = run_evaluation(Remote(usage), [_case("q1", must_contain=("alpha",))], FakeJudge(), usage)

    assert report.results[0].retrieval_hit is None


@pytest.mark.parametrize(
    ("excerpt", "phrase"),
    [
        ("The `pass` statement does nothing.", "The pass statement does nothing"),
        ("Words in **bold** and _italic_ text", "words in bold and italic text"),
        ("class instantiation invokes \\_\\_init\\_\\_()", "invokes __init__()"),
        ("two\n  lines    of text", "two lines of text"),
    ],
)
def test_an_expected_phrase_is_found_whatever_the_formatting_of_the_excerpt(excerpt: str, phrase: str) -> None:
    usage = UsageCounter()

    class Formatted(FakeSystem):
        def ask(self, case: Case) -> Outcome:
            return Outcome(answer="an answer", excerpts=(excerpt,))

    report = run_evaluation(Formatted(usage), [_case("q1", must_contain=(phrase,))], FakeJudge(), usage)

    assert report.results[0].retrieval_hit is True


def test_how_the_system_got_to_its_answer_is_kept_in_the_result() -> None:
    usage = UsageCounter()

    class Explaining(FakeSystem):
        def ask(self, case: Case) -> Outcome:
            return Outcome(
                answer="The source does not cover this question.",
                reason="The answer cites no excerpt.",
                steps=("Searched.", "Graded.", "Decided not to answer."),
                draft="A rejected answer.",
            )

    report = run_evaluation(Explaining(usage), [_case("q1")], FakeJudge(), usage)

    result = report.results[0]
    assert result.agent_reason == "The answer cites no excerpt."
    assert result.agent_steps == ("Searched.", "Graded.", "Decided not to answer.")
    assert result.agent_draft == "A rejected answer."


def test_a_system_that_does_not_explain_itself_leaves_those_fields_empty() -> None:
    usage = UsageCounter()

    report = run_evaluation(FakeSystem(usage), [_case("q1")], FakeJudge(), usage)

    result = report.results[0]
    assert (result.agent_reason, result.agent_draft, result.agent_steps) == (None, None, ())


def test_a_reply_of_the_judge_that_cannot_be_read_leaves_the_scores_empty_and_says_so() -> None:
    usage = UsageCounter()

    report = run_evaluation(FakeSystem(usage), [_case("q1")], FakeJudge("not json"), usage)

    result = report.results[0]
    assert (result.faithful, result.correct) == (None, None)
    assert result.reason == "The judge's reply could not be read."


def test_questions_that_were_already_answered_are_skipped() -> None:
    usage = UsageCounter()
    system = FakeSystem(usage)
    cases = [_case("q1"), _case("q2")]

    report = run_evaluation(system, cases, FakeJudge(), usage, skip={cases[0].key})

    assert system.asked == ["q2"]
    assert report.evaluated == 1


def test_a_run_stops_cleanly_when_its_limit_is_reached_and_keeps_what_it_has() -> None:
    usage = UsageCounter(max_calls={GROQ: 1})
    system = FakeSystem(usage)

    report = run_evaluation(system, [_case("q1"), _case("q2"), _case("q3")], FakeJudge(), usage)

    assert [result.question for result in report.results] == ["q1"]
    assert report.stopped_because == "The limit of 1 calls to groq has been reached."
    assert system.asked == ["q1", "q2"]


def test_a_run_stops_cleanly_when_a_service_stays_busy() -> None:
    usage = UsageCounter()
    system = FakeSystem(usage)
    system.failures["q2"] = ServiceBusyError("Groq is busy right now. Try again in a few seconds.")

    report = run_evaluation(system, [_case("q1"), _case("q2"), _case("q3")], FakeJudge(), usage)

    assert [result.question for result in report.results] == ["q1"]
    assert report.stopped_because == "Groq is busy right now. Try again in a few seconds."


def test_an_error_that_is_not_about_limits_is_not_hidden() -> None:
    usage = UsageCounter()
    system = FakeSystem(usage)
    system.failures["q1"] = RuntimeError("a bug in the system")

    with pytest.raises(RuntimeError, match="a bug in the system"):
        run_evaluation(system, [_case("q1")], FakeJudge(), usage)


# --- keeping the results ---


def test_the_results_are_saved_after_every_question(tmp_path: Path) -> None:
    out = tmp_path / "results" / "run.json"
    usage = UsageCounter()
    system = FakeSystem(usage)
    system.watch = out

    report = run_and_save(system, [_case("q1"), _case("q2"), _case("q3")], FakeJudge(), usage, out=out)

    assert system.saved_when_asked == [0, 1, 2]  # what was in the file when each question was asked
    assert len(load_results(out)) == 3
    assert [result.question for result in report.results] == ["q1", "q2", "q3"]
    assert [path.name for path in out.parent.iterdir()] == ["run.json"]  # no half-written file is left behind


def test_a_stopped_run_can_be_continued_without_asking_the_answered_questions_again(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    cases = [_case("q1"), _case("q2"), _case("q3")]
    first_usage = UsageCounter(max_calls={GROQ: 2})
    run_and_save(FakeSystem(first_usage), cases, FakeJudge(), first_usage, out=out)  # stops after two questions

    usage = UsageCounter()
    system = FakeSystem(usage)
    report = run_and_save(system, cases, FakeJudge(), usage, out=out, resume=True)

    assert system.asked == ["q3"]
    assert [result.question for result in report.results] == ["q1", "q2", "q3"]
    assert [result.question for result in load_results(out)] == ["q1", "q2", "q3"]
    assert report.evaluated == 1


def test_without_resume_an_existing_file_is_started_again(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    usage = UsageCounter()
    run_and_save(FakeSystem(usage), [_case("q1")], FakeJudge(), usage, out=out)
    system = FakeSystem(usage)

    report = run_and_save(system, [_case("q1"), _case("q2")], FakeJudge(), usage, out=out)

    assert system.asked == ["q1", "q2"]
    assert len(report.results) == 2


def test_a_small_first_run_can_be_limited_to_a_few_questions() -> None:
    usage = UsageCounter()
    system = FakeSystem(usage)

    report = run_and_save(system, [_case("q1"), _case("q2"), _case("q3")], FakeJudge(), usage, limit=2)

    assert system.asked == ["q1", "q2"]
    assert report.stopped_because is None


def test_nothing_is_written_when_no_file_is_given(tmp_path: Path) -> None:
    usage = UsageCounter()

    run_and_save(FakeSystem(usage), [_case("q1")], FakeJudge(), usage)

    assert list(tmp_path.iterdir()) == []


def test_results_that_cannot_be_read_stop_a_resumed_run(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    out.write_text('[{"unexpected": 1}]', encoding="utf-8")
    usage = UsageCounter()

    with pytest.raises(DatasetError, match="does not hold evaluation results"):
        run_and_save(FakeSystem(usage), [_case("q1")], FakeJudge(), usage, out=out, resume=True)


def test_results_survive_being_saved_and_read_again(tmp_path: Path) -> None:
    usage = UsageCounter()
    report = run_evaluation(FakeSystem(usage), [_case("q1"), _case("q2")], FakeJudge(), usage)
    path = tmp_path / "run.json"

    save_results(path, report.results)

    assert load_results(path) == report.results


def test_the_steps_of_a_result_survive_being_saved_and_read_again(tmp_path: Path) -> None:
    usage = UsageCounter()

    class Explaining(FakeSystem):
        def ask(self, case: Case) -> Outcome:
            return Outcome(answer="a", reason="why", steps=("one", "two"), draft="d")

    report = run_evaluation(Explaining(usage), [_case("q1")], FakeJudge(), usage)
    path = tmp_path / "run.json"

    save_results(path, report.results)

    (loaded,) = load_results(path)
    assert loaded == report.results[0]
    assert loaded.agent_steps == ("one", "two")  # a tuple again, not the list that JSON holds


def test_results_saved_before_the_steps_were_kept_can_still_be_read(tmp_path: Path) -> None:
    usage = UsageCounter()
    report = run_evaluation(FakeSystem(usage), [_case("q1")], FakeJudge(), usage)
    old = asdict(report.results[0])
    for name in ("agent_reason", "agent_draft", "agent_steps"):
        del old[name]
    path = _write(tmp_path / "old.json", [old])

    (loaded,) = load_results(path)

    assert loaded == report.results[0]


def test_a_file_that_is_not_a_list_of_results_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(DatasetError):
        load_results(path)


# --- the limits of a run ---


def test_the_limits_of_a_run_come_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "COHERE_API_KEY": "c",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "w",
        "GROQ_API_KEY": "g",
        "EVAL_MAX_COHERE_CALLS": "2",
        "EVAL_MAX_GROQ_CALLS": "3",
        "EVAL_MAX_GROQ_TOKENS": "50",
    }.items():
        monkeypatch.setenv(name, value)
    usage = evaluation_usage(Settings(_env_file=None))  # type: ignore[call-arg]

    usage.record_call(COHERE)
    usage.record_call(COHERE)
    with pytest.raises(BudgetExceededError):
        usage.record_call(COHERE)
    for _ in range(3):
        usage.record_call(GROQ)
    with pytest.raises(BudgetExceededError):
        usage.record_call(GROQ)
    with pytest.raises(BudgetExceededError):
        usage.record_tokens(GROQ, 51)
