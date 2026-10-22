"""Tests of the agent graph when it widens the excerpts, with scripted models and fakes instead of the services."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.graph import (
    EXPAND_ANSWER_CALLS,
    EXPAND_STEPS,
    AgentLimits,
    SourceAgent,
    next_step,
)
from chat_with_any_website_agent.pages import PageRead
from chat_with_any_website_agent.prompts import Grade
from chat_with_any_website_agent.retriever import SourceRetriever
from tests.test_graph import (
    ENOUGH,
    EXTRA_TEXT,
    FIRST_SEARCH,
    NOT_ENOUGH,
    REWRITTEN,
    SECOND_SEARCH,
    URL,
    FakeEmbedder,
    FakeReranker,
    FakeStore,
    ScriptedClient,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chat_with_any_website_agent.state import AgentState

QUESTION = "What does pass do?"
LIMITS = AgentLimits(max_steps=26, max_llm_calls=13, max_rewrites=2, max_expansions=1)
NEIGHBOUR = Document(
    page_content=EXTRA_TEXT,
    metadata={
        "url": URL,
        "title": "Control Flow",
        "heading": "pass Statements",
        "chunk_index": 2,
        "search_score": None,
        "rerank_score": None,
    },
)
ANSWER = "Pass is a null operation: nothing happens when it is executed [3]."


@dataclass
class FakeContext:
    """Gives the chunks that were configured for each widening, and records what it was asked."""

    results: list[list[Document]]
    calls: list[tuple[list[int], int]] = field(default_factory=list)

    def around(self, documents: Sequence[Document], reach: int = 1) -> list[Document]:
        self.calls.append(([int(d.metadata["chunk_index"]) for d in documents], reach))
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


@dataclass
class Setup:
    """An agent that can widen the excerpts, with the fakes behind it."""

    agent: SourceAgent
    store: FakeStore
    context: FakeContext
    grader: ScriptedClient
    answerer: ScriptedClient


def _setup(
    context_results: list[list[Document]],
    grader_replies: Sequence[str],
    answer_replies: Sequence[str] = (),
    *,
    limits: AgentLimits = LIMITS,
    searches: list[list] | None = None,  # type: ignore[type-arg]
    with_context: bool = True,
) -> Setup:
    store = FakeStore(searches if searches is not None else [FIRST_SEARCH])
    grader, answerer = ScriptedClient(*grader_replies), ScriptedClient(*answer_replies)
    retriever = SourceRetriever(store, FakeEmbedder(), FakeReranker())
    context = FakeContext(context_results)
    agent = SourceAgent(
        retriever,
        GroqChatModel(answerer),
        GroqChatModel(grader),
        limits,
        context=context if with_context else None,
    )
    return Setup(agent, store, context, grader, answerer)


# --- widening the excerpts when they are not enough ---


def test_excerpts_that_are_not_enough_are_widened_graded_again_and_then_answered_from() -> None:
    setup = _setup([[NEIGHBOUR]], [NOT_ENOUGH, ENOUGH], [ANSWER])

    result = setup.agent.ask(QUESTION, [URL])

    assert not result.abstained
    assert result.answer == ANSWER
    assert [citation.number for citation in result.citations] == [3]  # the added excerpt comes after the found ones
    assert result.steps == (
        "Read the question.",
        f'Searched for "{QUESTION}" and found 2 excerpt(s).',
        "Graded the excerpts: not enough (no definition).",
        "Added 1 excerpt(s) next to the best ones.",
        "Graded the excerpts: enough to answer.",
        "Wrote an answer.",
        "Checked the citations: they hold.",
    )
    assert result.llm_calls == 3  # two grades and the answer
    assert len(setup.store.searches) == 1  # nothing was searched again


def test_the_best_excerpts_are_widened_with_their_places_and_the_answer_sees_the_added_text() -> None:
    setup = _setup([[NEIGHBOUR]], [NOT_ENOUGH, ENOUGH], [ANSWER])

    setup.agent.ask(QUESTION, [URL])

    assert setup.context.calls == [([0, 1], 1)]  # the excerpts in order, and the nearest neighbours
    answer_request = setup.answerer.requests[0][0][-1]["content"]
    assert f"[3] Control Flow > pass Statements\n{EXTRA_TEXT}" in answer_request


def test_an_expansion_that_finds_nothing_is_not_graded_again_and_the_agent_goes_on() -> None:
    setup = _setup([[]], [NOT_ENOUGH, REWRITTEN, ENOUGH], [ANSWER], searches=[FIRST_SEARCH, SECOND_SEARCH])

    result = setup.agent.ask(QUESTION, [URL])

    assert "Looked next to the best excerpts: nothing to add." in result.steps
    assert result.steps.count("Graded the excerpts: not enough (no definition).") == 1  # no useless second grade
    assert 'Rewrote the search as "pass statement definition".' in result.steps  # it went on to the rewrite


def test_the_excerpts_are_widened_once_and_a_second_weak_grade_goes_on_to_the_rewrite() -> None:
    setup = _setup(
        [[NEIGHBOUR]],
        [NOT_ENOUGH, NOT_ENOUGH, REWRITTEN, ENOUGH],
        ["Pass does nothing [3]."],
        searches=[FIRST_SEARCH, SECOND_SEARCH],
    )

    result = setup.agent.ask(QUESTION, [URL])

    assert len(setup.context.calls) == 1
    steps = list(result.steps)
    assert steps.index("Added 1 excerpt(s) next to the best ones.") < steps.index(
        'Rewrote the search as "pass statement definition".'
    )


def test_a_second_expansion_looks_further_away_when_the_limit_allows_two() -> None:
    limits = AgentLimits(max_steps=40, max_llm_calls=20, max_rewrites=2, max_expansions=2)
    farther = Document(page_content="Farther text.", metadata={**NEIGHBOUR.metadata, "chunk_index": 4})
    setup = _setup(
        [[NEIGHBOUR], [farther]], [NOT_ENOUGH, NOT_ENOUGH, ENOUGH], ["Pass does nothing [3]."], limits=limits
    )

    result = setup.agent.ask(QUESTION, [URL])

    assert [reach for _, reach in setup.context.calls] == [1, 2]
    assert [indexes for indexes, _ in setup.context.calls] == [[0, 1], [0, 1, 2]]  # the second sees the first's
    assert result.steps.count("Added 1 excerpt(s) next to the best ones.") == 2


def test_excerpts_that_are_enough_at_the_first_grade_are_not_widened() -> None:
    setup = _setup([[NEIGHBOUR]], [ENOUGH], ["Pass does nothing [1]."])

    result = setup.agent.ask(QUESTION, [URL])

    assert setup.context.calls == []
    assert result.llm_calls == 2


def test_widening_is_switched_off_by_a_limit_of_zero() -> None:
    limits = AgentLimits(max_steps=26, max_llm_calls=13, max_rewrites=2, max_expansions=0)
    setup = _setup(
        [[NEIGHBOUR]],
        [NOT_ENOUGH, REWRITTEN, ENOUGH],
        ["Pass does nothing [1]."],
        limits=limits,
        searches=[FIRST_SEARCH, SECOND_SEARCH],
    )

    result = setup.agent.ask(QUESTION, [URL])

    assert setup.context.calls == []
    assert not any(step.startswith("Added") for step in result.steps)


def test_an_agent_without_a_source_of_context_never_widens_whatever_the_limit_says() -> None:
    setup = _setup(
        [[NEIGHBOUR]],
        [NOT_ENOUGH, REWRITTEN, ENOUGH],
        ["Pass does nothing [1]."],
        with_context=False,
        searches=[FIRST_SEARCH, SECOND_SEARCH],
    )

    result = setup.agent.ask(QUESTION, [URL])

    assert setup.context.calls == []
    assert 'Rewrote the search as "pass statement definition".' in result.steps


def test_the_pages_a_question_names_are_never_widened_but_refused_when_they_are_not_enough() -> None:
    class OnePage:
        def read(self, urls: Sequence[str], numbers: Sequence[int], query: str) -> PageRead:
            return PageRead(is_pdf=True, documents=(NEIGHBOUR,))

    store = FakeStore([FIRST_SEARCH])
    grader = ScriptedClient(NOT_ENOUGH)
    context = FakeContext([[NEIGHBOUR]])
    agent = SourceAgent(
        SourceRetriever(store, FakeEmbedder(), FakeReranker()),
        GroqChatModel(ScriptedClient()),
        GroqChatModel(grader),
        LIMITS,
        pages=OnePage(),
        context=context,
    )

    result = agent.ask("What does page 5 say?", [URL])

    assert result.abstained
    assert context.calls == []  # the answer has to come from the named page, so nothing is added from other places


# --- the route and the limits ---


def _graded(*, expansions: int, calls: int = 1, steps: int = 3, documents: int = 2, rewrites: int = 0) -> AgentState:
    return {
        "grade": Grade(sufficient=False, reason="no definition"),
        "llm_calls": calls,
        "steps": ["step"] * steps,
        "rewrites": rewrites,
        "expansions": expansions,
        "documents": [Document(page_content="text", metadata={})] * documents,
    }


def test_the_excerpts_are_widened_before_links_are_followed_and_before_a_retry() -> None:
    assert next_step(_graded(expansions=0), LIMITS) == "expand"
    assert next_step(_graded(expansions=1), LIMITS) == "rewrite"  # used up, so the next way goes on


def test_the_excerpts_are_widened_before_links_are_followed() -> None:
    limits = AgentLimits(max_steps=26, max_llm_calls=13, max_rewrites=2, max_link_hops=3, max_pages=5, max_expansions=1)
    state = _graded(expansions=0)
    state["frontier"] = [object()]  # type: ignore[list-item]

    assert next_step(state, limits) == "expand"
    state["expansions"] = 1
    assert next_step(state, limits) == "choose"


@pytest.mark.parametrize(
    ("calls", "steps", "expected"),
    [
        (13 - EXPAND_ANSWER_CALLS, 26 - EXPAND_STEPS - 3, "expand"),  # the new grade and the answer exactly fit
        (13 - EXPAND_ANSWER_CALLS + 1, 5, "rewrite"),  # one call short: no expansion, and no retry either
        (1, 26 - EXPAND_STEPS - 3 + 1, "rewrite"),  # one step short: no expansion
    ],
)
def test_the_excerpts_are_widened_only_when_the_new_grade_and_an_answer_fit_in_the_limits(
    calls: int, steps: int, expected: str
) -> None:
    state = _graded(expansions=0, calls=calls, steps=steps)

    route = next_step(state, LIMITS)

    assert (route == "expand") == (expected == "expand")


def test_nothing_is_widened_when_there_are_no_excerpts() -> None:
    assert next_step(_graded(expansions=0, documents=0), LIMITS) != "expand"


def test_an_expansion_is_counted_in_the_steps_and_calls_that_the_limits_need() -> None:
    limits = AgentLimits(max_steps=24, max_llm_calls=12, max_rewrites=2, max_link_hops=3, max_pages=5, max_expansions=1)

    assert limits.problems() == [
        "AGENT_MAX_STEPS is 24, but 3 link hop(s), 1 expansion(s), 2 rewrite(s) and an answer can need 26 steps, "
        "so hops and retries are skipped when they do not fit.",
        "AGENT_MAX_LLM_CALLS is 12, but 3 link hop(s), 1 expansion(s), 2 rewrite(s) and an answer can need 13 calls "
        "to the model, so hops and retries are skipped when they do not fit.",
    ]


def test_the_default_limits_have_room_for_every_way_including_one_expansion() -> None:
    limits = AgentLimits(max_steps=26, max_llm_calls=13, max_rewrites=2, max_link_hops=3, max_pages=5, max_expansions=1)

    assert limits.problems() == []
