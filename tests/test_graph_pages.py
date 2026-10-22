"""Tests of the agent graph when a question names a page of a PDF file, with scripted models and fakes."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.graph import AgentLimits, SourceAgent, next_step
from chat_with_any_website_agent.pages import PageRead
from chat_with_any_website_agent.prompts import NOT_COVERED_MESSAGE, Grade
from chat_with_any_website_agent.retriever import SourceRetriever
from tests.test_graph import (
    ENOUGH,
    FIRST_SEARCH,
    NOT_ENOUGH,
    FakeEmbedder,
    FakeReranker,
    FakeStore,
    ScriptedClient,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

PAPER = "https://arxiv.example/pdf/1706.03762"
LIMITS = AgentLimits(max_steps=24, max_llm_calls=12, max_rewrites=2)
QUESTION = "What does page 5 say about the heads?"
HEADS = Document(
    page_content="Multi-head attention uses h = 8 parallel attention heads.",
    metadata={
        "url": PAPER,
        "title": "Attention",
        "heading": "Page 5",
        "chunk_index": 8,
        "search_score": None,
        "rerank_score": None,
    },
)
CITED_ANSWER = "Multi-head attention uses 8 parallel attention heads [1]."
SEARCHED_ANSWER = "The pass statement does nothing [1]."  # what the fake search finds


@dataclass
class FakePages:
    """Reads the pages it was given, and records what it was asked."""

    result: PageRead
    calls: list[tuple[list[str], list[int], str]] = field(default_factory=list)

    def read(self, urls: Sequence[str], numbers: Sequence[int], query: str) -> PageRead:
        self.calls.append((list(urls), list(numbers), query))
        return self.result


@dataclass
class Setup:
    """An agent that can read pages, with the fakes behind it."""

    agent: SourceAgent
    store: FakeStore
    pages: FakePages
    grader: ScriptedClient
    answerer: ScriptedClient


def _setup(
    result: PageRead,
    grader_replies: Sequence[str] = (),
    answer_replies: Sequence[str] = (),
    *,
    with_reader: bool = True,
) -> Setup:
    store = FakeStore([FIRST_SEARCH])
    grader, answerer = ScriptedClient(*grader_replies), ScriptedClient(*answer_replies)
    retriever = SourceRetriever(store, FakeEmbedder(), FakeReranker())
    pages = FakePages(result)
    agent = SourceAgent(
        retriever, GroqChatModel(answerer), GroqChatModel(grader), LIMITS, pages=pages if with_reader else None
    )
    return Setup(agent, store, pages, grader, answerer)


FOUND = PageRead(is_pdf=True, documents=(HEADS,))


# --- a question that names a page ---


def test_the_named_page_is_read_and_the_answer_comes_from_it_without_a_search() -> None:
    setup = _setup(FOUND, [ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION, [PAPER])

    assert not result.abstained
    assert result.answer == CITED_ANSWER
    assert [(citation.number, citation.url, citation.heading) for citation in result.citations] == [
        (1, PAPER, "Page 5")
    ]
    assert result.steps == (
        "Read the question.",
        "Read page 5 and found 1 excerpt(s).",
        "Graded the excerpts: enough to answer.",
        "Wrote an answer.",
        "Checked the citations: they hold.",
    )
    assert setup.store.searches == []  # nothing was searched by meaning
    assert setup.pages.calls == [([PAPER], [5], QUESTION)]
    assert result.llm_calls == 2


def test_the_pages_are_read_for_a_question_in_arabic_too() -> None:
    setup = _setup(FOUND, [ENOUGH], [CITED_ANSWER])

    setup.agent.ask("ماذا تقول الصفحة 5 عن الرؤوس؟", [PAPER])

    assert [numbers for _, numbers, _ in setup.pages.calls] == [[5]]


def test_when_the_pages_are_not_enough_the_agent_does_not_search_elsewhere_and_says_so() -> None:
    setup = _setup(FOUND, [NOT_ENOUGH])

    result = setup.agent.ask(QUESTION, [PAPER])

    assert result.abstained
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.reason == "no definition"
    assert setup.store.searches == []  # a search of the whole document could answer about another page
    assert len(setup.grader.requests) == 1  # no rewrite was asked for
    assert result.llm_calls == 1
    assert result.steps[-1] == "Decided not to answer: no definition"


def test_a_page_without_text_is_refused_at_once_with_a_message_that_says_why() -> None:
    setup = _setup(PageRead(is_pdf=True, missing=(40,)))

    result = setup.agent.ask("What does page 40 say?", [PAPER])

    message = f"{NOT_COVERED_MESSAGE} This PDF file has no text for page 40. The document may have fewer pages."
    assert result.abstained
    assert result.answer.startswith("The source does not cover this question")  # it is recognized as a refusal
    assert result.answer == message  # and the person is told what is wrong, not only that nothing was found
    assert result.reason == message
    assert result.citations == ()
    assert result.llm_calls == 0  # no model was asked
    assert setup.grader.requests == []
    assert setup.answerer.requests == []
    assert result.steps == ("Read the question.", f"Read page 40: {message}", f"Decided not to answer: {message}")


def test_the_note_of_the_document_explains_why_a_page_has_no_text() -> None:
    note = (
        "No text can be selected on page 4, which may be scans or images, so a question about them cannot be answered."
    )
    setup = _setup(PageRead(is_pdf=True, missing=(4,), note=note))

    result = setup.agent.ask("What does page 4 say?", [PAPER])

    assert result.answer == f"{NOT_COVERED_MESSAGE} This PDF file has no text for page 4. {note}"


def test_a_missing_page_is_said_in_the_steps_when_another_named_page_has_text() -> None:
    setup = _setup(PageRead(is_pdf=True, documents=(HEADS,), missing=(40,)), [ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask("What do pages 5 and 40 say?", [PAPER])

    assert not result.abstained
    assert "Read pages 5 and 40 and found 1 excerpt(s); no text was found for page 40." in result.steps


def test_several_pages_are_named_in_the_steps() -> None:
    setup = _setup(FOUND, [ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask("What do pages 3 to 5 say?", [PAPER])

    assert "Read pages 3, 4 and 5 and found 1 excerpt(s)." in result.steps
    assert setup.pages.calls[0][1] == [3, 4, 5]


# --- a question that is not about a page of a PDF file ---


def test_a_number_in_a_question_about_a_source_that_is_not_a_pdf_is_not_a_page() -> None:
    setup = _setup(PageRead(is_pdf=False), [ENOUGH], [SEARCHED_ANSWER])

    result = setup.agent.ask(QUESTION, ["https://docs.example/tutorial"])

    assert "The question names page 5, but no source is a PDF file: searched as usual." in result.steps
    assert len(setup.store.searches) == 1  # it was searched as any question is
    assert not result.abstained


def test_after_a_source_that_is_not_a_pdf_a_weak_search_can_still_be_improved_by_a_rewrite() -> None:
    rewritten = '{"query": "multi-head attention heads"}'
    setup = _setup(PageRead(is_pdf=False), [NOT_ENOUGH, rewritten, ENOUGH], [SEARCHED_ANSWER])
    setup.store.results = [FIRST_SEARCH, [*FIRST_SEARCH]]

    result = setup.agent.ask(QUESTION, ["https://docs.example/tutorial"])

    assert 'Rewrote the search as "multi-head attention heads".' in result.steps


def test_a_question_without_a_page_never_reads_pages() -> None:
    setup = _setup(FOUND, [ENOUGH], ["The pass statement does nothing [1]."])

    setup.agent.ask("What does the pass statement do?", [PAPER])

    assert setup.pages.calls == []
    assert len(setup.store.searches) == 1


def test_an_agent_that_cannot_read_pages_treats_a_page_number_as_part_of_the_question() -> None:
    setup = _setup(FOUND, [ENOUGH], ["The pass statement does nothing [1]."], with_reader=False)

    result = setup.agent.ask(QUESTION, [PAPER])

    assert setup.pages.calls == []
    assert len(setup.store.searches) == 1
    assert "Read the question." in result.steps


def test_when_every_source_is_searched_no_page_is_read() -> None:
    setup = _setup(FOUND, [ENOUGH], ["The pass statement does nothing [1]."])

    setup.agent.ask(QUESTION)  # no sources given: the agent does not know which one to read

    assert setup.pages.calls == []
    assert len(setup.store.searches) == 1


# --- the route after the grade ---


def _graded(*, enough: bool, pages: list[int]) -> dict[str, object]:
    return {
        "grade": Grade(sufficient=enough),
        "llm_calls": 1,
        "steps": ["step"] * 3,
        "rewrites": 0,
        "pages": pages,
    }


@pytest.mark.parametrize(
    ("enough", "pages", "expected"),
    [
        (True, [5], "generate"),
        (True, [], "generate"),
        (False, [5], "abstain"),  # the answer has to come from the named pages
        (False, [], "rewrite"),
    ],
)
def test_the_route_after_the_grade_keeps_to_the_named_pages(enough: bool, pages: list[int], expected: str) -> None:
    state = _graded(enough=enough, pages=pages)

    assert next_step(state, LIMITS) == expected  # type: ignore[arg-type]
