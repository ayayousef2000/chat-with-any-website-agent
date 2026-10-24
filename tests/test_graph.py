"""Tests of the agent graph, with scripted models and fakes instead of Groq, Cohere and Weaviate."""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from langgraph.errors import GraphRecursionError

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.citations import NO_CITATION_PROBLEM
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import ServiceBusyError
from chat_with_any_website_agent.graph import (
    LIMIT_REASON,
    NO_ANSWER_REASON,
    NO_EXCERPTS_REASON,
    NOTHING_NEW_NOTE,
    STEP_LIMIT_REASON,
    AgentAnswer,
    AgentLimits,
    SourceAgent,
    next_step,
)
from chat_with_any_website_agent.groq_client import ChatReply
from chat_with_any_website_agent.prompts import NOT_COVERED_MESSAGE, Grade
from chat_with_any_website_agent.reranking import RankedDocument
from chat_with_any_website_agent.retriever import SourceRetriever
from chat_with_any_website_agent.vector_store import RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chat_with_any_website_agent.state import AgentState

URL = "https://docs.example/control"
QUESTION = "What does pass do?"
PASS_TEXT = "The pass statement does nothing. It is used when a statement is required but no action is needed."
SET_TEXT = "A set is an unordered collection with no duplicate elements."

ENOUGH = '{"sufficient": true, "reason": "stated"}'
NOT_ENOUGH = '{"sufficient": false, "reason": "no definition"}'
REWRITTEN = '{"query": "pass statement definition"}'
CITED_ANSWER = "The pass statement does nothing [1]."
LIMITS = AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=2)


def _chunk(index: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        url=URL, title="Control Flow", heading="pass Statements", text=text, chunk_index=index, score=0.5
    )


EXTRA_TEXT = "Pass is a null operation: nothing happens when it is executed."
MORE_TEXT = "It is useful as a placeholder when code has to be written later."
FIRST_SEARCH = [_chunk(0, PASS_TEXT), _chunk(1, SET_TEXT)]
SECOND_SEARCH = [*FIRST_SEARCH, _chunk(2, EXTRA_TEXT)]  # one excerpt that the first search did not find
THIRD_SEARCH = [*SECOND_SEARCH, _chunk(3, MORE_TEXT)]


class FakeStore:
    """Returns the chunks of each search in turn, the last ones again at the end, and records the searches."""

    def __init__(self, results: list[list[RetrievedChunk]]) -> None:
        self.results = results
        self.searches: list[tuple[str, Sequence[str] | None]] = []

    def search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float, urls: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        number = len(self.searches)
        self.searches.append((query, urls))
        return self.results[min(number, len(self.results) - 1)]


class FakeEmbedder:
    """Embeds every question the same way, or fails."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    def embed_query(self, text: str) -> list[float]:
        if self.error is not None:
            raise self.error
        return [0.1]


class FakeReranker:
    """Keeps the order of the search."""

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RankedDocument]:
        return [RankedDocument(index=index, score=1.0 - index / 10) for index in range(min(top_n, len(documents)))]


class ScriptedClient:
    """Answers with the replies it was given, in order, and records its requests."""

    model = "model-x"

    def __init__(self, *replies: str) -> None:
        self.replies = deque(replies)
        self.requests: list[tuple[list[Any], dict[str, Any]]] = []

    def complete(self, messages: Sequence[Any], **options: Any) -> ChatReply:
        self.requests.append((list(messages), options))
        if not self.replies:
            raise AssertionError("The model was called more often than the test expected.")
        return ChatReply(text=self.replies.popleft(), prompt_tokens=10, completion_tokens=5, key_number=1)


@dataclass
class Setup:
    """An agent with the fakes behind it."""

    agent: SourceAgent
    store: FakeStore
    grader: ScriptedClient
    answerer: ScriptedClient


def _setup(
    grader_replies: Sequence[str],
    answer_replies: Sequence[str] = (),
    chunks: list[RetrievedChunk] | None = None,
    limits: AgentLimits = LIMITS,
    embedder: FakeEmbedder | None = None,
    searches: list[list[RetrievedChunk]] | None = None,
) -> Setup:
    """An agent whose searches find ``searches`` in turn, or else always ``chunks``, or else the first two chunks."""
    store = FakeStore(searches if searches is not None else [FIRST_SEARCH if chunks is None else chunks])
    grader, answerer = ScriptedClient(*grader_replies), ScriptedClient(*answer_replies)
    retriever = SourceRetriever(store, embedder or FakeEmbedder(), FakeReranker())
    agent = SourceAgent(retriever, GroqChatModel(answerer), GroqChatModel(grader), limits)
    return Setup(agent, store, grader, answerer)


def _state(*, enough: bool, calls: int = 1, steps: int = 3, rewrites: int = 0) -> AgentState:
    return {"grade": Grade(sufficient=enough), "llm_calls": calls, "steps": ["step"] * steps, "rewrites": rewrites}


# --- a question that the source answers ---


def test_a_question_that_the_excerpts_answer_gets_a_cited_answer() -> None:
    setup = _setup([ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION, [URL])

    assert not result.abstained
    assert result.answer == CITED_ANSWER
    assert result.reason is None
    assert [citation.number for citation in result.citations] == [1]
    assert result.sources == (URL,)
    assert result.excerpts == (PASS_TEXT, SET_TEXT)
    assert result.llm_calls == 2
    assert result.steps == (
        "Read the question.",
        f'Searched for "{QUESTION}" and found 2 excerpt(s).',
        "Graded the excerpts: enough to answer.",
        "Wrote an answer.",
        "Checked the citations: they hold.",
    )
    assert setup.store.searches == [(QUESTION, [URL])]


def test_the_models_are_asked_for_json_when_grading_and_see_the_numbered_excerpts() -> None:
    setup = _setup([ENOUGH], [CITED_ANSWER])

    setup.agent.ask(QUESTION)

    grade_messages, grade_options = setup.grader.requests[0]
    answer_messages, answer_options = setup.answerer.requests[0]
    assert grade_options == {"response_format": {"type": "json_object"}, "temperature": 0}
    assert answer_options == {"temperature": 0}
    assert f"[1] Control Flow > pass Statements\n{PASS_TEXT}" in grade_messages[-1]["content"]
    assert f"Question: {QUESTION}" in answer_messages[-1]["content"]


def test_the_question_is_tidied_and_every_source_is_searched_when_none_are_given() -> None:
    setup = _setup([ENOUGH], [CITED_ANSWER])

    setup.agent.ask("  What   does pass\n do?  ")

    assert setup.store.searches == [(QUESTION, None)]


def test_each_source_that_the_question_gives_is_searched_on_its_own() -> None:
    other = "https://docs.example/other"
    setup = _setup([ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION, [URL, other])

    assert not result.abstained
    assert setup.store.searches == [(QUESTION, [URL]), (QUESTION, [other])]


def test_a_single_given_source_is_searched_once_with_all_its_pages() -> None:
    setup = _setup([ENOUGH], [CITED_ANSWER])

    setup.agent.ask(QUESTION, [URL])

    assert setup.store.searches == [(QUESTION, [URL])]


def test_a_blank_question_is_refused() -> None:
    setup = _setup([])

    with pytest.raises(ValueError, match="empty"):
        setup.agent.ask("   ")


# --- a question that needs a better search ---


def test_a_search_that_finds_too_little_is_made_again_with_a_rewritten_query() -> None:
    setup = _setup([NOT_ENOUGH, REWRITTEN, ENOUGH], [CITED_ANSWER], searches=[FIRST_SEARCH, SECOND_SEARCH])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert [query for query, _ in setup.store.searches] == [QUESTION, "pass statement definition"]
    assert result.llm_calls == 4
    assert 'Rewrote the search as "pass statement definition".' in result.steps
    assert 'Searched for "pass statement definition" and found 3 excerpt(s), 1 of them new.' in result.steps
    rewrite_request = setup.grader.requests[1][0][-1]["content"]
    assert "Previous query: " + QUESTION in rewrite_request
    assert "What was missing: no definition" in rewrite_request


def test_a_rewrite_that_cannot_be_read_searches_with_the_question_again() -> None:
    setup = _setup([NOT_ENOUGH, "no idea", ENOUGH], [CITED_ANSWER], searches=[FIRST_SEARCH, SECOND_SEARCH])

    result = setup.agent.ask(QUESTION)

    assert [query for query, _ in setup.store.searches] == [QUESTION, QUESTION]
    assert "The new query could not be read, so the search will use the question again." in result.steps
    assert not result.abstained


def test_a_verdict_that_cannot_be_read_goes_on_to_the_answer_and_its_check() -> None:
    setup = _setup(["not json"], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert result.llm_calls == 2


# --- a question that the source does not answer ---


def test_a_source_that_never_has_enough_ends_in_an_abstention_after_the_retries() -> None:
    setup = _setup(
        [NOT_ENOUGH, REWRITTEN, NOT_ENOUGH, REWRITTEN, NOT_ENOUGH], searches=[FIRST_SEARCH, SECOND_SEARCH, THIRD_SEARCH]
    )

    result = setup.agent.ask(QUESTION)

    assert result == AgentAnswer(
        answer=NOT_COVERED_MESSAGE,
        abstained=True,
        reason="no definition",
        excerpts=(PASS_TEXT, SET_TEXT, EXTRA_TEXT, MORE_TEXT),
        steps=result.steps,
        llm_calls=5,
    )
    assert len(setup.store.searches) == 3
    assert setup.answerer.requests == []
    assert result.steps[-1] == "Decided not to answer: no definition"


def test_a_search_that_finds_nothing_new_ends_the_question_without_grading_the_same_excerpts_again() -> None:
    setup = _setup([NOT_ENOUGH, REWRITTEN])  # the second search finds the same two excerpts as the first

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == f"no definition. {NOTHING_NEW_NOTE}"
    assert result.llm_calls == 2  # the first grade and the rewrite, and nothing more
    assert len(setup.store.searches) == 2
    assert setup.answerer.requests == []
    assert result.steps[-2:] == (
        'Searched for "pass statement definition" and found 2 excerpt(s), 0 of them new.',
        f"Decided not to answer: no definition. {NOTHING_NEW_NOTE}",
    )


def test_a_search_that_finds_only_some_of_what_was_found_before_finds_nothing_new() -> None:
    setup = _setup([NOT_ENOUGH, REWRITTEN], searches=[FIRST_SEARCH, [_chunk(0, PASS_TEXT)]])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == f"no definition. {NOTHING_NEW_NOTE}"
    assert result.llm_calls == 2


def test_an_excerpt_found_by_an_earlier_search_is_not_new_even_when_the_search_before_missed_it() -> None:
    first, second, third = [_chunk(0, PASS_TEXT)], [_chunk(1, SET_TEXT)], [_chunk(0, PASS_TEXT), _chunk(1, SET_TEXT)]
    setup = _setup([NOT_ENOUGH, REWRITTEN, NOT_ENOUGH, REWRITTEN], searches=[first, second, third])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert len(setup.store.searches) == 3  # the second search found a new excerpt, the third only known ones
    assert result.llm_calls == 4  # two grades and two rewrites; the third search was not graded
    assert result.reason == f"no definition. {NOTHING_NEW_NOTE}"


def test_a_search_that_finds_nothing_is_not_graded_by_the_model() -> None:
    setup = _setup([], chunks=[], limits=AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=0))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == NO_EXCERPTS_REASON
    assert result.llm_calls == 0
    assert setup.grader.requests == []


def test_a_search_that_finds_nothing_twice_ends_the_question_after_one_retry() -> None:
    setup = _setup([REWRITTEN], chunks=[])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == f"{NO_EXCERPTS_REASON} {NOTHING_NEW_NOTE}"
    assert result.llm_calls == 1  # only the rewrite: nothing was found, so nothing was graded
    assert len(setup.store.searches) == 2


@pytest.mark.parametrize("reply", ["NOT_COVERED", "not_covered.", "**NOT_COVERED**", '"NOT_COVERED"\n'])
def test_a_model_that_finds_no_answer_in_the_excerpts_ends_in_an_abstention(reply: str) -> None:
    setup = _setup([ENOUGH], [reply])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.reason == NO_ANSWER_REASON
    assert "The model found no answer in the excerpts." in result.steps


UNCITED_ANSWER = "The pass statement does nothing."


def test_an_answer_without_a_citation_is_corrected_once_and_then_given() -> None:
    setup = _setup([ENOUGH], [UNCITED_ANSWER, CITED_ANSWER])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert result.answer == CITED_ANSWER
    assert result.draft is None
    assert result.llm_calls == 3
    assert result.steps[-4:] == (
        "Wrote an answer.",
        f"Checked the citations: {NO_CITATION_PROBLEM}",
        "Asked the model to correct its answer.",
        "Checked the citations: they hold.",
    )
    repair_request = setup.answerer.requests[1][0][-1]["content"]
    assert f"Your previous answer:\n{UNCITED_ANSWER}" in repair_request
    assert f"Why it was rejected: {NO_CITATION_PROBLEM}" in repair_request


INVENTED_FIGURE = "The pass statement does nothing and was added in 1991 [1]."
INVENTED_PROBLEM = "The answer states 1991, which the excerpts it cites do not contain."


def test_an_answer_with_a_figure_that_the_excerpts_do_not_state_is_corrected_with_the_reason() -> None:
    setup = _setup([ENOUGH], [INVENTED_FIGURE, CITED_ANSWER])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert result.answer == CITED_ANSWER
    assert f"Checked the citations: {INVENTED_PROBLEM}" in result.steps
    repair_request = setup.answerer.requests[1][0][-1]["content"]
    assert f"Your previous answer:\n{INVENTED_FIGURE}" in repair_request
    assert f"Why it was rejected: {INVENTED_PROBLEM}" in repair_request


def test_a_model_that_cannot_support_its_figure_ends_in_an_abstention_with_the_rejected_draft() -> None:
    setup = _setup([ENOUGH], [INVENTED_FIGURE, "It was added in 1992 [1]."])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.draft == "It was added in 1992 [1]."
    assert result.reason == "The answer states 1992, which the excerpts it cites do not contain."


def test_a_figure_of_the_question_may_be_repeated_in_the_answer() -> None:
    setup = _setup([ENOUGH], ["In Python 3.14 the pass statement does nothing [1]."])

    result = setup.agent.ask("What does the pass statement do in Python 3.14?")

    assert not result.abstained
    assert len(setup.answerer.requests) == 1  # no correction was needed


def test_a_citation_in_other_brackets_is_accepted_and_written_with_plain_brackets() -> None:
    answer = f"The pass statement does nothing {chr(0x3010)}1{chr(0x3011)}."
    setup = _setup([ENOUGH], [answer])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert result.answer == CITED_ANSWER
    assert [citation.number for citation in result.citations] == [1]
    assert len(setup.answerer.requests) == 1  # no correction was needed
    assert result.llm_calls == 2


def test_an_answer_that_is_rejected_again_is_not_given_and_the_rejected_text_is_kept() -> None:
    setup = _setup([ENOUGH], [UNCITED_ANSWER, "Still no citation here."])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.reason == NO_CITATION_PROBLEM
    assert result.draft == "Still no citation here."
    assert result.citations == ()
    assert result.sources == ()
    assert len(setup.answerer.requests) == 2  # the answer and one correction, never more


def test_an_answer_that_cites_an_excerpt_that_does_not_exist_is_corrected_too() -> None:
    setup = _setup([ENOUGH], ["The pass statement does nothing [7].", CITED_ANSWER])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert "The answer cites excerpt 7, which does not exist." in setup.answerer.requests[1][0][-1]["content"]


def test_a_correction_that_says_the_excerpts_hold_no_answer_ends_in_an_abstention_without_a_draft() -> None:
    setup = _setup([ENOUGH], [UNCITED_ANSWER, "NOT_COVERED"])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == NO_ANSWER_REASON
    assert result.draft is None


def test_no_correction_is_asked_when_the_calls_to_the_model_are_used_up() -> None:
    setup = _setup([ENOUGH], [UNCITED_ANSWER], limits=AgentLimits(max_steps=12, max_llm_calls=2, max_rewrites=2))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == NO_CITATION_PROBLEM
    assert result.draft == UNCITED_ANSWER
    assert len(setup.answerer.requests) == 1


def test_no_correction_is_asked_when_it_cannot_finish_within_the_step_limit() -> None:
    setup = _setup([ENOUGH], [UNCITED_ANSWER], limits=AgentLimits(max_steps=7, max_llm_calls=8, max_rewrites=2))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == NO_CITATION_PROBLEM
    assert len(setup.answerer.requests) == 1


# --- the limits ---


def test_the_defaults_are_enough_for_the_longest_path_without_reaching_the_step_limit() -> None:
    # Two rewrites, a model that has an answer without a citation, and then the abstention: the longest path.
    setup = _setup(
        [NOT_ENOUGH, REWRITTEN, NOT_ENOUGH, REWRITTEN, ENOUGH],
        ["The pass statement does nothing."],
        searches=[FIRST_SEARCH, SECOND_SEARCH, THIRD_SEARCH],
    )

    result = setup.agent.ask(QUESTION)

    assert len(result.steps) == LIMITS.max_steps
    assert result.reason == NO_CITATION_PROBLEM
    assert result.llm_calls == 6


def test_no_retry_is_made_when_none_is_allowed() -> None:
    setup = _setup([NOT_ENOUGH], limits=AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=0))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == "no definition"
    assert len(setup.store.searches) == 1


def test_no_answer_is_written_when_the_calls_to_the_model_are_used_up() -> None:
    setup = _setup([ENOUGH], limits=AgentLimits(max_steps=12, max_llm_calls=1, max_rewrites=2))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == LIMIT_REASON
    assert setup.answerer.requests == []


def test_the_graph_stops_by_itself_when_the_step_limit_is_too_small_for_an_answer() -> None:
    setup = _setup([ENOUGH], limits=AgentLimits(max_steps=1, max_llm_calls=8, max_rewrites=2))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert setup.answerer.requests == []


def test_the_recursion_limit_of_the_graph_is_a_last_safeguard(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _setup([])

    def too_deep(*args: Any, **kwargs: Any) -> None:
        raise GraphRecursionError("limit")

    monkeypatch.setattr(setup.agent._graph, "invoke", too_deep)

    result = setup.agent.ask(QUESTION)

    assert result == AgentAnswer(answer=NOT_COVERED_MESSAGE, abstained=True, reason=STEP_LIMIT_REASON)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (_state(enough=True), "generate"),
        (_state(enough=True, steps=9), "generate"),  # 9 + 3 = 12, exactly the limit
        (_state(enough=True, steps=10), "abstain"),
        (_state(enough=True, calls=8), "abstain"),
        (_state(enough=False), "rewrite"),
        (_state(enough=False, steps=6), "rewrite"),  # 6 + 6 = 12, exactly the limit
        (_state(enough=False, steps=7), "abstain"),
        (_state(enough=False, calls=5), "rewrite"),  # three calls are left, enough for rewrite, grade and generate
        (_state(enough=False, calls=6), "abstain"),
        (_state(enough=False, rewrites=1), "rewrite"),
        (_state(enough=False, rewrites=2), "abstain"),
    ],
)
def test_what_follows_the_grade_depends_on_the_verdict_and_on_what_is_left_of_the_limits(
    state: AgentState, expected: str
) -> None:
    assert next_step(state, LIMITS) == expected


def test_link_hops_are_counted_in_the_limits_that_are_needed() -> None:
    limits = AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=2, max_link_hops=3, max_pages=5)

    assert limits.problems() == [
        "AGENT_MAX_STEPS is 12, but 3 link hop(s), 2 rewrite(s) and an answer can need 24 steps, "
        "so hops and retries are skipped when they do not fit.",
        "AGENT_MAX_LLM_CALLS is 8, but 3 link hop(s), 2 rewrite(s) and an answer can need 12 calls to the model, "
        "so hops and retries are skipped when they do not fit.",
    ]


def test_the_hops_are_also_ended_by_the_pages_limit() -> None:
    assert AgentLimits(24, 12, 2, max_link_hops=3, max_pages=1).hops == 1
    assert AgentLimits(24, 12, 2, max_link_hops=1, max_pages=5).hops == 1
    assert AgentLimits(24, 12, 2, max_link_hops=3, max_pages=0).hops == 0
    assert AgentLimits(24, 12, 2).hops == 0  # without hops the limits are those of the retries alone


def test_hops_that_cannot_happen_need_no_room() -> None:
    assert AgentLimits(12, 8, 2, max_link_hops=0, max_pages=5).problems() == []
    assert AgentLimits(12, 8, 2, max_link_hops=3, max_pages=0).problems() == []


def test_the_default_limits_have_no_problem() -> None:
    assert LIMITS.problems() == []


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (AgentLimits(max_steps=8, max_llm_calls=8, max_rewrites=2), ["AGENT_MAX_STEPS is 8, but 2 rewrite(s)"]),
        (AgentLimits(max_steps=12, max_llm_calls=5, max_rewrites=2), ["AGENT_MAX_LLM_CALLS is 5, but 2 rewrite(s)"]),
        (AgentLimits(max_steps=9, max_llm_calls=8, max_rewrites=1), []),  # exactly what one rewrite needs
        (AgentLimits(max_steps=6, max_llm_calls=2, max_rewrites=0), []),  # no rewrites, so no retry to make room for
    ],
)
def test_limits_that_are_too_small_for_the_allowed_retries_are_named(limits: AgentLimits, expected: list[str]) -> None:
    notes = limits.problems()

    assert len(notes) == len(expected)
    for note, start in zip(notes, expected, strict=True):
        assert note.startswith(start)


def test_both_limits_can_be_too_small_at_once() -> None:
    notes = AgentLimits(max_steps=3, max_llm_calls=1, max_rewrites=2).problems()

    assert [note.split(" is ")[0] for note in notes] == ["AGENT_MAX_STEPS", "AGENT_MAX_LLM_CALLS"]
    assert "can need 12 steps" in notes[0]
    assert "need 6 calls to the model" in notes[1]


def test_the_limits_come_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "w")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("AGENT_MAX_LLM_CALLS", "5")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert AgentLimits.from_settings(settings) == AgentLimits(
        max_steps=24, max_llm_calls=5, max_rewrites=2, max_link_hops=3, max_pages=5
    )


# --- failures of the services ---


def test_a_failure_of_a_service_reaches_the_caller() -> None:
    error = ServiceBusyError("Cohere is busy.", retry_after=30.0)
    setup = _setup([], embedder=FakeEmbedder(error))

    with pytest.raises(ServiceBusyError, match="busy"):
        setup.agent.ask(QUESTION)


# --- the answer is written from the excerpts that the grade named ---


def _enough_from(*numbers: int) -> str:
    """The reply of a grader that finds the excerpts enough and names the ones the answer needs."""
    return '{"sufficient": true, "reason": "stated", "useful": [' + ", ".join(str(number) for number in numbers) + "]}"


def test_the_answer_is_written_and_checked_from_the_excerpts_that_the_grade_named() -> None:
    setup = _setup([_enough_from(2)], ["A set has no duplicate elements [1]."])

    result = setup.agent.ask("What is a set?", [URL])

    request = setup.answerer.requests[0][0][-1]["content"]
    assert SET_TEXT in request
    assert PASS_TEXT not in request
    assert request.count("[1] ") == 1  # the named excerpt is the first the model sees, and the answer cites it so
    assert not result.abstained
    assert [citation.number for citation in result.citations] == [1]
    assert result.excerpts == (SET_TEXT,)  # what the answer is based on, not what the search found
    assert "Graded the excerpts: enough to answer, from 1 of 2." in result.steps


@pytest.mark.parametrize(
    ("reply", "kept"),
    [
        (_enough_from(), (PASS_TEXT, SET_TEXT)),  # none named: all are used, as before
        (_enough_from(0, 3, 7), (PASS_TEXT, SET_TEXT)),  # numbers that are not excerpts name nothing
        (_enough_from(2, 2, 9), (SET_TEXT,)),  # a number is counted once, and a wrong one is ignored
        (_enough_from(2, 1), (PASS_TEXT, SET_TEXT)),  # the order of the search is kept, not the order named
        (_enough_from(1, 2), (PASS_TEXT, SET_TEXT)),
    ],
)
def test_the_numbers_the_grade_names_are_checked_before_they_are_used(reply: str, kept: tuple[str, ...]) -> None:
    setup = _setup([reply], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION, [URL])

    assert result.excerpts == kept


def test_a_correction_is_asked_from_the_same_excerpts_as_the_answer() -> None:
    setup = _setup([_enough_from(2)], [INVENTED_FIGURE, "A set has no duplicate elements [1]."])

    result = setup.agent.ask("What is a set?", [URL])

    assert not result.abstained
    repair_request = setup.answerer.requests[1][0][-1]["content"]
    assert SET_TEXT in repair_request
    assert PASS_TEXT not in repair_request


def test_the_excerpts_named_by_a_grade_that_was_not_enough_are_not_kept_for_the_next_answer() -> None:
    weak_with_a_number = '{"sufficient": false, "reason": "no definition", "useful": [1]}'
    setup = _setup(
        [weak_with_a_number, REWRITTEN, _enough_from(3)],
        ["Pass is a null operation [1]."],
        searches=[FIRST_SEARCH, SECOND_SEARCH],
    )

    result = setup.agent.ask(QUESTION, [URL])

    request = setup.answerer.requests[0][0][-1]["content"]
    assert EXTRA_TEXT in request
    assert PASS_TEXT not in request  # number 1 of the weak grade named nothing; number 3 of the second grade did
    assert result.excerpts == (EXTRA_TEXT,)
