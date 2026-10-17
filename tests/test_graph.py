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


class FakeStore:
    """Returns the chunks it was given and records the questions and the sources it was asked about."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.searches: list[tuple[str, Sequence[str] | None]] = []

    def search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float, urls: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append((query, urls))
        return self.chunks


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
) -> Setup:
    store = FakeStore([_chunk(0, PASS_TEXT), _chunk(1, SET_TEXT)] if chunks is None else chunks)
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


def test_a_blank_question_is_refused() -> None:
    setup = _setup([])

    with pytest.raises(ValueError, match="empty"):
        setup.agent.ask("   ")


# --- a question that needs a better search ---


def test_a_search_that_finds_too_little_is_made_again_with_a_rewritten_query() -> None:
    setup = _setup([NOT_ENOUGH, REWRITTEN, ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION)

    assert not result.abstained
    assert [query for query, _ in setup.store.searches] == [QUESTION, "pass statement definition"]
    assert result.llm_calls == 4
    assert 'Rewrote the search as "pass statement definition".' in result.steps
    rewrite_request = setup.grader.requests[1][0][-1]["content"]
    assert "Previous query: " + QUESTION in rewrite_request
    assert "What was missing: no definition" in rewrite_request


def test_a_rewrite_that_cannot_be_read_searches_with_the_question_again() -> None:
    setup = _setup([NOT_ENOUGH, "no idea", ENOUGH], [CITED_ANSWER])

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
    setup = _setup([NOT_ENOUGH, REWRITTEN, NOT_ENOUGH, REWRITTEN, NOT_ENOUGH])

    result = setup.agent.ask(QUESTION)

    assert result == AgentAnswer(
        answer=NOT_COVERED_MESSAGE,
        abstained=True,
        reason="no definition",
        excerpts=(PASS_TEXT, SET_TEXT),
        steps=result.steps,
        llm_calls=5,
    )
    assert len(setup.store.searches) == 3
    assert setup.answerer.requests == []
    assert result.steps[-1] == "Decided not to answer: no definition"


def test_a_search_that_finds_nothing_is_not_graded_by_the_model() -> None:
    setup = _setup([], chunks=[], limits=AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=0))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == NO_EXCERPTS_REASON
    assert result.llm_calls == 0
    assert setup.grader.requests == []


def test_a_search_that_keeps_finding_nothing_is_retried_until_the_retries_are_used() -> None:
    setup = _setup([REWRITTEN, REWRITTEN], chunks=[])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == NO_EXCERPTS_REASON
    assert result.llm_calls == 2
    assert len(setup.store.searches) == 3


@pytest.mark.parametrize("reply", ["NOT_COVERED", "not_covered.", "**NOT_COVERED**", '"NOT_COVERED"\n'])
def test_a_model_that_finds_no_answer_in_the_excerpts_ends_in_an_abstention(reply: str) -> None:
    setup = _setup([ENOUGH], [reply])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.reason == NO_ANSWER_REASON
    assert "The model found no answer in the excerpts." in result.steps


def test_an_answer_without_a_citation_is_not_given() -> None:
    setup = _setup([ENOUGH], ["The pass statement does nothing."])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.reason == NO_CITATION_PROBLEM
    assert result.citations == ()
    assert result.sources == ()


def test_an_answer_that_cites_an_excerpt_that_does_not_exist_is_not_given() -> None:
    setup = _setup([ENOUGH], ["The pass statement does nothing [7]."])

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert result.reason == "The answer cites excerpt 7, which does not exist."


# --- the limits ---


def test_the_defaults_are_enough_for_the_longest_path_without_reaching_the_step_limit() -> None:
    # Two rewrites, a model that has an answer without a citation, and then the abstention: the longest path.
    setup = _setup([NOT_ENOUGH, REWRITTEN, NOT_ENOUGH, REWRITTEN, ENOUGH], ["The pass statement does nothing."])

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


def test_the_limits_come_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "w")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("AGENT_MAX_LLM_CALLS", "5")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert AgentLimits.from_settings(settings) == AgentLimits(max_steps=12, max_llm_calls=5, max_rewrites=2)


# --- failures of the services ---


def test_a_failure_of_a_service_reaches_the_caller() -> None:
    error = ServiceBusyError("Cohere is busy.", retry_after=30.0)
    setup = _setup([], embedder=FakeEmbedder(error))

    with pytest.raises(ServiceBusyError, match="busy"):
        setup.agent.ask(QUESTION)
