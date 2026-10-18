"""Scoring a question-answering system on a set of questions.

A system is anything that can load a source and answer a question about it: the plain pipeline, a version that
crawls the whole site first, the original app behind its web address, or the agent. Every question is answered by the
system and judged by a model against a reference answer. Metrics per question:

- ``retrieval_hit``: the excerpts the answer was based on contain at least one expected phrase (no model involved).
- ``faithful``: a judge confirms that every claim in the answer is supported by those excerpts.
- ``correct``: a judge confirms that the answer agrees with the reference answer. For a question that the sources
  cannot answer, it confirms that the answer says so.

The cost of each question is recorded too (calls to Cohere and Groq, Groq tokens and seconds). A run stops cleanly
when the call or token limit it was given is reached, and the results are saved after every question, so that a run
can be continued the next day.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from chat_with_any_website_agent.errors import BudgetExceededError, DatasetError, ServiceBusyError
from chat_with_any_website_agent.usage import COHERE, GROQ, UsageCounter

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence
    from pathlib import Path

    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.groq_client import ChatReply

DEFAULT_CATEGORY = "single_page"
DEFAULT_LANGUAGE = "en"
NOT_ANSWERABLE = "NOT ANSWERABLE FROM THE SOURCES"

# A judge that gives an empty or unreadable reply is asked once more before the question is left unscored.
JUDGE_ATTEMPTS = 2

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system that must answer only from the excerpts
of its sources.
Reply with a JSON object: {"faithful": true|false|null, "correct": true|false, "reason": "<one sentence>"}.

- faithful: every factual claim in the answer is supported by the excerpts. An answer that only says the sources
  do not cover the question is faithful. If no excerpts are given, use null.
- correct: if the question is answerable, the answer agrees with the reference answer in substance. The reference
  answer is the truth: an answer that says the sources do not cover the question is NOT correct for an answerable
  question, even when the excerpts really lack the information (that is a failure of the search).
  If the question is NOT answerable from the sources, the answer must clearly say the sources do not cover it
  and must not invent an answer."""


@dataclass(frozen=True)
class Turn:
    """An earlier question of a conversation and the answer it got."""

    question: str
    answer: str


@dataclass(frozen=True)
class Case:
    """One question to evaluate.

    Attributes:
        url: The source the question is about.
        question: The question.
        answerable: ``False`` when the sources cannot answer it, so the right answer is that they do not cover it.
        reference: The right answer. Required when the question is answerable.
        must_contain: Phrases of which at least one should be in the excerpts the answer is based on.
        history: The earlier turns of the conversation, for a follow-up question.
        category: The kind of question, for the summary (for example ``multi_page`` or ``unanswerable``).
        language: The language of the question, for the summary.
        sources: Further sources the answer needs, besides ``url``.
    """

    url: str
    question: str
    answerable: bool = True
    reference: str | None = None
    must_contain: tuple[str, ...] = ()
    history: tuple[Turn, ...] = ()
    category: str = DEFAULT_CATEGORY
    language: str = DEFAULT_LANGUAGE
    sources: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """What identifies the question in a results file."""
        return f"{self.url}\n{self.question}"


@dataclass(frozen=True)
class Outcome:
    """What a system answered.

    Attributes:
        answer: The answer, as the person would see it.
        excerpts: The texts the answer is based on, which the judge checks the answer against.
        sources: The addresses of the sources that were used.
        reason: Why the system did not answer from the source, when it did not.
        steps: What the system did to get there, one line for each step.
        draft: An answer that the system wrote and then refused, when that is why it did not answer.
    """

    answer: str
    excerpts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    reason: str | None = None
    steps: tuple[str, ...] = ()
    draft: str | None = None


class System(Protocol):
    """A question-answering system that can be evaluated."""

    def prepare(self, url: str) -> None:
        """Get ready to answer questions about a source, for example by loading it. Called once for each source."""

    def ask(self, case: Case) -> Outcome:
        """Answer a question."""

    def close(self) -> None:
        """Let go of what the system holds, such as connections."""


class ChatClient(Protocol):
    """What the judge needs from a model client, which ``GroqClient`` provides."""

    def complete(self, messages: Sequence[Any], **options: Any) -> ChatReply:
        """Send messages to the model and return its reply."""


@dataclass(frozen=True)
class Verdict:
    """What the judge decided about an answer.

    Attributes:
        faithful: Whether every claim is supported by the excerpts, or ``None`` when there were no excerpts.
        correct: Whether the answer agrees with the reference.
        reason: One sentence of why.
    """

    faithful: bool | None
    correct: bool
    reason: str


@dataclass(frozen=True)
class Result:
    """The scores and the cost of one evaluated question.

    The ``agent_`` fields say how the system got to its answer, so that a wrong answer can be understood from the
    results file. They are empty for a system that does not report them and in results saved before they existed.
    """

    url: str
    question: str
    category: str
    language: str
    answerable: bool
    retrieval_hit: bool | None
    faithful: bool | None
    correct: bool | None
    reason: str
    answer: str
    seconds: float
    cohere_calls: int
    groq_calls: int
    groq_tokens: int
    agent_reason: str | None = None
    agent_draft: str | None = None
    agent_steps: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """What identifies the question in a results file."""
        return f"{self.url}\n{self.question}"


@dataclass
class RunReport:
    """The outcome of a run.

    Attributes:
        results: All the results, those of earlier runs that were continued first.
        evaluated: How many questions this run answered.
        stopped_because: Why the run ended before the last question, or ``None`` when it finished.
    """

    results: list[Result] = field(default_factory=list)
    evaluated: int = 0
    stopped_because: str | None = None


# --- the question set ---


def load_cases(path: Path) -> list[Case]:
    """Read a question set and check it.

    The file holds a list of pages, each with a ``url`` and its ``questions``. A question has a ``question`` and, when
    it is answerable, a ``reference``. It may have ``must_contain``, ``history``, ``category``, ``language`` and
    ``sources``.

    Args:
        path: The JSON file.

    Returns:
        The questions in the order of the file.

    Raises:
        DatasetError: If the file cannot be read or does not follow the format. The message says where.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{path}: the file cannot be read as JSON ({type(exc).__name__}).") from exc
    if not isinstance(data, list):
        raise DatasetError(f"{path}: the file must hold a list of pages.")
    cases: list[Case] = []
    for page_number, page in enumerate(data, start=1):
        where = f"{path}, page {page_number}"
        if not isinstance(page, dict):
            raise DatasetError(f"{where}: a page must be an object with a url and questions.")
        url = _text(page.get("url"), "url", where)
        questions = page.get("questions")
        if not isinstance(questions, list) or not questions:
            raise DatasetError(f"{where}: a page needs a non-empty list of questions.")
        for number, item in enumerate(questions, start=1):
            cases.append(_case(item, url, f"{where}, question {number}"))
    seen: set[str] = set()
    for case in cases:
        if case.key in seen:
            raise DatasetError(f"{path}: the question {case.question!r} is asked twice about {case.url}.")
        seen.add(case.key)
    return cases


def _text(value: Any, name: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{where}: {name} must be a non-empty text.")
    return value.strip()


def _texts(value: Any, name: str, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DatasetError(f"{where}: {name} must be a list of texts.")
    return tuple(_text(item, name, where) for item in value)


def _turn(turn: Any, where: str) -> Turn:
    if not isinstance(turn, dict):
        raise DatasetError(f"{where}: every turn of the history must be an object.")
    return Turn(
        question=_text(turn.get("question"), "history question", where),
        answer=_text(turn.get("answer"), "history answer", where),
    )


def _case(item: Any, url: str, where: str) -> Case:
    if not isinstance(item, dict):
        raise DatasetError(f"{where}: a question must be an object.")
    answerable = item.get("answerable", True)
    if not isinstance(answerable, bool):
        raise DatasetError(f"{where}: answerable must be true or false.")
    reference = item.get("reference")
    if answerable:
        reference = _text(reference, "reference (an answerable question needs one)", where)
    history_items = item.get("history") or []
    if not isinstance(history_items, list):
        raise DatasetError(f"{where}: history must be a list of turns.")
    history = tuple(_turn(turn, where) for turn in history_items)
    return Case(
        url=url,
        question=_text(item.get("question"), "question", where),
        answerable=answerable,
        reference=reference if isinstance(reference, str) else None,
        must_contain=_texts(item.get("must_contain"), "must_contain", where),
        history=history,
        category=_text(item.get("category", DEFAULT_CATEGORY), "category", where),
        language=_text(item.get("language", DEFAULT_LANGUAGE), "language", where),
        sources=_texts(item.get("sources"), "sources", where),
    )


# --- judging ---


def parse_verdict(text: str) -> Verdict | None:
    """Read the judge's reply.

    Args:
        text: The reply, which should be a JSON object with ``faithful``, ``correct`` and ``reason``.

    Returns:
        The verdict, or ``None`` when the reply cannot be read, so that it is not counted as a wrong answer.
    """
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("correct"), bool):
        return None
    faithful = data.get("faithful")
    if faithful is not None and not isinstance(faithful, bool):
        return None
    return Verdict(faithful=faithful, correct=data["correct"], reason=str(data.get("reason", "")))


def judge_answer(judge: ChatClient, case: Case, outcome: Outcome) -> Verdict | None:
    """Ask the judge model whether an answer is faithful to its excerpts and correct.

    Args:
        judge: The model client of the judge.
        case: The question with its reference answer.
        outcome: What the system answered.

    Returns:
        The verdict, or ``None`` when the judge's reply could not be read, even after asking again once.
    """
    reference = case.reference if case.answerable else NOT_ANSWERABLE
    excerpts = "\n\n".join(f"[{number}] {text}" for number, text in enumerate(outcome.excerpts, start=1))
    user_message = (
        f"Question: {case.question}\n\nReference answer: {reference}\n\n"
        f"Retrieved excerpts:\n{excerpts or '(none)'}\n\nSystem answer: {outcome.answer}"
    )
    messages = [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user_message}]
    for _ in range(JUDGE_ATTEMPTS):
        reply = judge.complete(messages, response_format={"type": "json_object"}, temperature=0)
        verdict = parse_verdict(reply.text)
        if verdict is not None:
            return verdict
    return None


# --- running ---


def evaluation_usage(settings: Settings) -> UsageCounter:
    """A counter with the limits of an evaluation run, so that a run can never use up a quota.

    Args:
        settings: The settings, with ``EVAL_MAX_COHERE_CALLS``, ``EVAL_MAX_GROQ_CALLS`` and ``EVAL_MAX_GROQ_TOKENS``.

    Returns:
        A counter to give to the system, the judge and ``run_evaluation``, so that one limit covers all of them.
    """
    return UsageCounter(
        max_calls={COHERE: settings.eval_max_cohere_calls, GROQ: settings.eval_max_groq_calls},
        max_tokens={GROQ: settings.eval_max_groq_tokens},
    )


def run_evaluation(
    system: System,
    cases: Sequence[Case],
    judge: ChatClient,
    usage: UsageCounter,
    *,
    skip: Collection[str] = (),
    on_result: Callable[[Result], None] | None = None,
) -> RunReport:
    """Answer and judge the questions one after the other.

    The run ends early, with what it has, when a limit of calls or tokens is reached or when a service stays busy.

    Args:
        system: The system to evaluate.
        cases: The questions.
        judge: The model client of the judge.
        usage: Counts the calls and tokens of the run. The system and the judge should be given the same counter, so
            that one limit covers all of them.
        skip: The keys of the questions that were already answered.
        on_result: Called after every question, for example to save the results.

    Returns:
        The results of this run, and why it stopped if it did.
    """
    report = RunReport()
    prepared: set[str] = set()
    try:
        for case in cases:
            if case.key in skip:
                continue
            if case.url not in prepared:
                system.prepare(case.url)
                prepared.add(case.url)
            result = _evaluate(system, judge, usage, case)
            report.results.append(result)
            report.evaluated += 1
            if on_result is not None:
                on_result(result)
    except (BudgetExceededError, ServiceBusyError) as error:
        report.stopped_because = str(error)
    return report


def _evaluate(system: System, judge: ChatClient, usage: UsageCounter, case: Case) -> Result:
    calls_before, tokens_before = usage.calls, usage.tokens
    started = time.monotonic()
    outcome = system.ask(case)
    seconds = time.monotonic() - started
    calls_after, tokens_after = usage.calls, usage.tokens
    verdict = judge_answer(judge, case, outcome)
    return Result(
        url=case.url,
        question=case.question,
        category=case.category,
        language=case.language,
        answerable=case.answerable,
        retrieval_hit=_retrieval_hit(case, outcome),
        faithful=verdict.faithful if verdict else None,
        correct=verdict.correct if verdict else None,
        reason=verdict.reason if verdict else "The judge's reply could not be read.",
        answer=outcome.answer,
        seconds=round(seconds, 3),
        cohere_calls=calls_after.get(COHERE, 0) - calls_before.get(COHERE, 0),
        groq_calls=calls_after.get(GROQ, 0) - calls_before.get(GROQ, 0),
        groq_tokens=tokens_after.get(GROQ, 0) - tokens_before.get(GROQ, 0),
        agent_reason=outcome.reason,
        agent_draft=outcome.draft,
        agent_steps=outcome.steps,
    )


# Markdown marks that the text of a page may have around words: `code`, *emphasis*, _emphasis_ and escapes.
_FORMATTING = str.maketrans("", "", "`*_\\")


def _plain(text: str) -> str:
    """Lower-case a text without its Markdown marks and with its white space made single, to compare phrases."""
    return " ".join(text.translate(_FORMATTING).casefold().split())


def _retrieval_hit(case: Case, outcome: Outcome) -> bool | None:
    """Whether the excerpts contain an expected phrase; ``None`` when that cannot be told.

    The comparison ignores capitals, white space and Markdown marks, so that "The pass statement does nothing"
    is found in "The `pass` statement does nothing".
    """
    if not case.answerable or not case.must_contain or not outcome.excerpts:
        return None
    text = _plain("\n".join(outcome.excerpts))
    return any(_plain(phrase) in text for phrase in case.must_contain)


def run_and_save(
    system: System,
    cases: Sequence[Case],
    judge: ChatClient,
    usage: UsageCounter,
    *,
    out: Path | None = None,
    resume: bool = False,
    limit: int | None = None,
) -> RunReport:
    """Run an evaluation and keep the results in a file, saved after every question.

    Args:
        system: The system to evaluate.
        cases: The questions.
        judge: The model client of the judge.
        usage: Counts the calls and tokens of the run.
        out: The results file. Nothing is saved when it is ``None``.
        resume: Continue the results already in ``out``: the questions in it are not asked again.
        limit: Answer at most this many more questions, for a small first run.

    Returns:
        All the results, those found in the file first, and why the run stopped if it did.

    Raises:
        DatasetError: If the results file cannot be read.
    """
    previous = load_results(out) if resume and out is not None and out.exists() else []
    known = {result.key for result in previous}
    pending = [case for case in cases if case.key not in known]
    if limit is not None:
        pending = pending[:limit]
    saved = list(previous)

    def keep(result: Result) -> None:
        saved.append(result)
        if out is not None:
            save_results(out, saved)

    report = run_evaluation(system, pending, judge, usage, on_result=keep)
    report.results = saved
    return report


# --- the results file ---


def save_results(path: Path, results: Sequence[Result]) -> None:
    """Write the results as JSON, replacing the file in one step so that a stop never leaves half a file.

    Args:
        path: The results file.
        results: The results to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_results(path: Path) -> list[Result]:
    """Read the results of an earlier run.

    Args:
        path: The results file.

    Returns:
        The results in the order of the file.

    Raises:
        DatasetError: If the file cannot be read or does not hold results.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_result(item) for item in data]
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetError(f"{path}: the file does not hold evaluation results ({type(exc).__name__}).") from exc


def _result(item: Any) -> Result:
    """One saved result. A result saved before the ``agent_`` fields existed has them empty."""
    fields = dict(item)
    fields["agent_steps"] = tuple(fields.get("agent_steps", ()))  # JSON has no tuples
    return Result(**fields)
