"""Summarizing the results of an evaluation run, overall and for each category of question."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from chat_with_any_website_agent.evaluation import Result

OVERALL = "all"


@dataclass(frozen=True)
class GroupSummary:
    """The scores and the average cost of a group of questions.

    A rate is ``None`` when no question of the group could be scored for it.

    Attributes:
        questions: How many questions the group has.
        unjudged: How many answers the judge's reply could not be read for. They are left out of the rates.
        retrieval_hit_rate: The share of answerable questions whose excerpts held an expected phrase.
        faithfulness: The share of answers whose claims the excerpts support.
        correctness: The share of answers that agree with the reference.
        phrase_rate: The share of answerable questions with expected phrases whose answer states all of them.
        abstention: The share of unanswerable questions that were answered by saying the sources do not cover them.
        seconds: The average time to answer a question.
        cohere_calls: The average calls to Cohere for a question.
        groq_calls: The average calls to Groq for a question.
        groq_tokens: The average tokens Groq used for a question.
    """

    questions: int
    unjudged: int
    retrieval_hit_rate: float | None
    faithfulness: float | None
    correctness: float | None
    phrase_rate: float | None
    abstention: float | None
    seconds: float
    cohere_calls: float
    groq_calls: float
    groq_tokens: float


def summarize(results: Sequence[Result]) -> dict[str, GroupSummary]:
    """Summarize the results overall and for each category.

    Args:
        results: The results of a run.

    Returns:
        ``"all"`` first, then each category in alphabetical order. Empty when there are no results.
    """
    if not results:
        return {}
    summaries = {OVERALL: _summarize_group(results)}
    for category in sorted({result.category for result in results}):
        summaries[category] = _summarize_group([result for result in results if result.category == category])
    return summaries


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _summarize_group(results: Sequence[Result]) -> GroupSummary:
    return GroupSummary(
        questions=len(results),
        unjudged=sum(result.correct is None for result in results),
        retrieval_hit_rate=_rate([r.retrieval_hit for r in results if r.retrieval_hit is not None]),
        faithfulness=_rate([r.faithful for r in results if r.faithful is not None]),
        correctness=_rate([r.correct for r in results if r.correct is not None]),
        phrase_rate=_rate([r.phrases_found for r in results if r.phrases_found is not None]),
        abstention=_rate([r.correct for r in results if not r.answerable and r.correct is not None]),
        seconds=_average([r.seconds for r in results]),
        cohere_calls=_average([r.cohere_calls for r in results]),
        groq_calls=_average([r.groq_calls for r in results]),
        groq_tokens=_average([r.groq_tokens for r in results]),
    )


def _percent(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.0%}"


def format_summary(summaries: Mapping[str, GroupSummary]) -> str:
    """Write the summaries as a table of text.

    Args:
        summaries: The summaries, as ``summarize`` returns them.

    Returns:
        A table with a row for each group, and a note when some answers could not be judged.
    """
    if not summaries:
        return "No results."
    header = (
        f"{'group':<20}{'n':>5}{'hit':>6}{'faithful':>10}{'correct':>9}{'phrases':>9}{'abstain':>9}"
        f"{'s/q':>8}{'groq/q':>8}{'tokens/q':>10}{'cohere/q':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, group in summaries.items():
        lines.append(
            f"{name:<20}{group.questions:>5}{_percent(group.retrieval_hit_rate):>6}"
            f"{_percent(group.faithfulness):>10}{_percent(group.correctness):>9}{_percent(group.phrase_rate):>9}"
            f"{_percent(group.abstention):>9}"
            f"{group.seconds:>8.1f}{group.groq_calls:>8.1f}{group.groq_tokens:>10.0f}{group.cohere_calls:>10.1f}"
        )
    unjudged = summaries[OVERALL].unjudged if OVERALL in summaries else 0
    if unjudged:
        lines.append(f"\n{unjudged} answer(s) could not be judged and are left out of the rates.")
    return "\n".join(lines)
