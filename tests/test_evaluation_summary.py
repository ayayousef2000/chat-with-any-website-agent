"""Tests of the summary of an evaluation run."""

from typing import Any

import pytest

from chat_with_any_website_agent.evaluation import Result
from chat_with_any_website_agent.evaluation_summary import OVERALL, format_summary, summarize


def _result(question: str = "q", **kwargs: Any) -> Result:
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
        "seconds": 2.0,
        "cohere_calls": 1,
        "groq_calls": 2,
        "groq_tokens": 1000,
    }
    values.update(kwargs)
    return Result(**values)


def test_there_is_nothing_to_summarize_without_results() -> None:
    assert summarize([]) == {}
    assert format_summary({}) == "No results."


def test_the_summary_has_an_overall_row_and_a_row_for_each_category() -> None:
    results = [_result("a"), _result("b", category="multi_page"), _result("c", category="multi_page")]

    summaries = summarize(results)

    assert list(summaries) == [OVERALL, "multi_page", "single_page"]
    assert [summaries[name].questions for name in summaries] == [3, 2, 1]


def test_rates_count_only_the_questions_that_can_be_scored_for_them() -> None:
    results = [
        _result("a", retrieval_hit=True, faithful=True, correct=True),
        _result("b", retrieval_hit=False, faithful=False, correct=False),
        _result("c", retrieval_hit=None, faithful=None, correct=True),  # no excerpts to check
        _result("d", retrieval_hit=True, faithful=True, correct=None),  # the judge's reply could not be read
    ]

    overall = summarize(results)[OVERALL]

    assert overall.retrieval_hit_rate == pytest.approx(2 / 3)
    assert overall.faithfulness == pytest.approx(2 / 3)
    assert overall.correctness == pytest.approx(2 / 3)
    assert overall.unjudged == 1
    assert overall.questions == 4


def test_abstention_counts_only_the_questions_the_sources_cannot_answer() -> None:
    results = [
        _result("a", answerable=False, correct=True),
        _result("b", answerable=False, correct=False),
        _result("c", answerable=True, correct=False),
    ]

    assert summarize(results)[OVERALL].abstention == 0.5


def test_a_rate_nobody_could_be_scored_for_is_empty() -> None:
    overall = summarize([_result(retrieval_hit=None, faithful=None, correct=None, answerable=True)])[OVERALL]

    assert (overall.retrieval_hit_rate, overall.faithfulness, overall.correctness, overall.abstention) == (
        None,
        None,
        None,
        None,
    )


def test_the_cost_is_averaged_over_all_the_questions() -> None:
    results = [
        _result("a", seconds=1.0, cohere_calls=0, groq_calls=2, groq_tokens=500),
        _result("b", seconds=3.0, cohere_calls=2, groq_calls=4, groq_tokens=1500),
    ]

    overall = summarize(results)[OVERALL]

    assert (overall.seconds, overall.cohere_calls, overall.groq_calls, overall.groq_tokens) == (2.0, 1.0, 3.0, 1000.0)


def test_the_table_has_a_row_for_each_group_and_marks_what_could_not_be_scored() -> None:
    results = [_result("a"), _result("b", category="multi_page", correct=None, faithful=None, retrieval_hit=None)]

    table = format_summary(summarize(results))

    lines = table.splitlines()
    columns = ["group", "n", "hit", "faithful", "correct", "abstain", "s/q", "groq/q", "tokens/q", "cohere/q"]
    assert lines[0].split() == columns
    assert lines[2].split()[:6] == ["all", "2", "100%", "100%", "100%", "-"]
    assert lines[3].split()[:6] == ["multi_page", "1", "-", "-", "-", "-"]
    assert "1 answer(s) could not be judged" in table
