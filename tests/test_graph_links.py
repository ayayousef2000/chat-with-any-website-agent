"""Tests of the agent graph when it follows links, with scripted models and fakes instead of the services."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.graph import (
    FOLLOW_CALLS,
    FOLLOW_STEPS,
    MAX_CANDIDATES,
    MAX_LINKS_PER_HOP,
    AgentLimits,
    SourceAgent,
    next_step,
)
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.navigation import FollowResult
from chat_with_any_website_agent.prompts import Grade
from chat_with_any_website_agent.retriever import SourceRetriever
from chat_with_any_website_agent.vector_store import RetrievedChunk
from tests.test_graph import ENOUGH, NOT_ENOUGH, REWRITTEN, FakeEmbedder, FakeReranker, ScriptedClient

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from chat_with_any_website_agent.state import AgentState

START = "https://docs.example/tutorial"
CONTROL = "https://docs.example/tutorial/control"
DATA = "https://docs.example/tutorial/data"
ERRORS = "https://docs.example/tutorial/errors"
QUESTION = "What does pass do?"
LIMITS = AgentLimits(max_steps=24, max_llm_calls=12, max_rewrites=2, max_link_hops=3, max_pages=5)

CONTENTS = "Contents: control flow, data structures, errors and exceptions."
PASS_TEXT = "The pass statement does nothing. It is used when a statement is required but no action is needed."
LIST_TEXT = "A list holds items in order, and append adds an item to its end."
CITED_ANSWER = "The pass statement does nothing [2]."

PAGES = {
    START: [CONTENTS],
    CONTROL: [PASS_TEXT],
    DATA: [LIST_TEXT],
}
LINKS = {
    START: (Link(CONTROL, "Control flow"), Link(DATA, "Data structures")),
    CONTROL: (Link(START, "Tutorial"), Link(ERRORS, "Errors")),
    DATA: (),
}


def choose(*numbers: int) -> str:
    """The reply of the model that chooses the links with these numbers."""
    return '{"links": [' + ", ".join(str(number) for number in numbers) + "]}"


class PagedStore:
    """Finds the chunks of the pages that are searched, one chunk for each text of the page."""

    def __init__(self, pages: dict[str, list[str]]) -> None:
        self.pages = pages
        self.searches: list[Sequence[str] | None] = []

    def search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float, urls: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append(urls)
        wanted = list(self.pages) if urls is None else urls
        return [
            RetrievedChunk(url=url, title=f"Page {url[-7:]}", heading="", text=text, chunk_index=index, score=0.5)
            for url in wanted
            for index, text in enumerate(self.pages.get(url, []))
        ]


@dataclass
class FakeLinks:
    """Knows the links of the pages and what opening each link comes to; records both."""

    links: Mapping[str, tuple[Link, ...]]
    refused: dict[str, str] = field(default_factory=dict)  # address -> why it cannot be opened
    asked: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)

    def links_of(self, url: str) -> tuple[Link, ...]:
        self.asked.append(url)
        return self.links.get(url, ())

    def follow(self, url: str) -> FollowResult:
        self.opened.append(url)
        if url in self.refused:
            return FollowResult(url=url, loaded=False, note=self.refused[url])
        return FollowResult(url=url, loaded=True, title="A page", chunk_count=1)


@dataclass
class Setup:
    """An agent that can follow links, with the fakes behind it."""

    agent: SourceAgent
    store: PagedStore
    links: FakeLinks
    grader: ScriptedClient
    answerer: ScriptedClient


def _setup(
    grader_replies: Sequence[str],
    answer_replies: Sequence[str] = (),
    *,
    limits: AgentLimits = LIMITS,
    links: Mapping[str, tuple[Link, ...]] | None = None,
    refused: dict[str, str] | None = None,
    pages: dict[str, list[str]] | None = None,
) -> Setup:
    store = PagedStore(PAGES if pages is None else pages)
    fake_links = FakeLinks(LINKS if links is None else links, refused or {})
    grader, answerer = ScriptedClient(*grader_replies), ScriptedClient(*answer_replies)
    retriever = SourceRetriever(store, FakeEmbedder(), FakeReranker())
    agent = SourceAgent(retriever, GroqChatModel(answerer), GroqChatModel(grader), limits, fake_links)
    return Setup(agent, store, fake_links, grader, answerer)


# --- following a link to the answer ---


def test_a_question_that_the_first_page_does_not_answer_is_answered_from_a_link() -> None:
    setup = _setup([NOT_ENOUGH, choose(1), ENOUGH], [CITED_ANSWER])

    result = setup.agent.ask(QUESTION, [START])

    assert not result.abstained
    assert result.answer == CITED_ANSWER
    assert result.followed == (CONTROL,)
    assert result.sources == (CONTROL,)  # the answer cites the page that was followed
    assert result.llm_calls == 4  # grade, choose, grade, answer
    assert result.steps == (
        "Read the question.",
        f'Searched for "{QUESTION}" and found 1 excerpt(s).',
        "Graded the excerpts: not enough (no definition).",
        f'Chose links to open: "Control flow" ({CONTROL}).',
        f"Followed links: opened {CONTROL}.",
        f'Searched for "{QUESTION}" and found 2 excerpt(s).',
        "Graded the excerpts: enough to answer.",
        "Wrote an answer.",
        "Checked the citations: they hold.",
    )
    assert setup.links.opened == [CONTROL]
    assert setup.store.searches == [[START], [START, CONTROL]]  # all the pages are searched after the hop


def test_a_question_that_the_first_page_answers_follows_nothing() -> None:
    setup = _setup([ENOUGH], ["The pass statement does nothing [1]."])
    setup.store.pages = {START: [PASS_TEXT]}

    result = setup.agent.ask(QUESTION, [START])

    assert not result.abstained
    assert result.followed == ()
    assert setup.links.opened == []


def test_the_model_sees_the_links_as_numbers_with_their_text_and_path() -> None:
    setup = _setup([NOT_ENOUGH, choose()], limits=AgentLimits(24, 12, 0, 3, 5))

    setup.agent.ask(QUESTION, [START])

    messages, options = setup.grader.requests[1]
    assert options == {"response_format": {"type": "json_object"}, "temperature": 0}
    system, human = messages[0]["content"], messages[-1]["content"]
    assert "untrusted" in system
    assert f"Question: {QUESTION}" in human
    assert "What was missing: no definition" in human
    assert "1. Control flow - /tutorial/control\n2. Data structures - /tutorial/data" in human
    assert "docs.example" not in human  # every link is on the same site, so the site is not repeated


def test_only_numbers_of_the_list_can_be_chosen_and_each_only_once() -> None:
    setup = _setup([NOT_ENOUGH, choose(0, 9, 2, 2, -1), ENOUGH], ["The pass statement does nothing [2]."])
    setup.store.pages = {START: [CONTENTS], DATA: [PASS_TEXT]}

    result = setup.agent.ask(QUESTION, [START])

    assert setup.links.opened == [DATA]
    assert result.followed == (DATA,)


def test_at_most_the_allowed_number_of_links_is_opened_in_one_hop() -> None:
    many = {START: tuple(Link(f"https://docs.example/p{number}", f"Page {number}") for number in range(5))}
    setup = _setup([NOT_ENOUGH, choose(1, 2, 3, 4), NOT_ENOUGH], links=many, limits=AgentLimits(24, 12, 0, 1, 5))

    setup.agent.ask(QUESTION, [START])

    assert setup.links.opened == ["https://docs.example/p0", "https://docs.example/p1"]
    assert len(setup.links.opened) == MAX_LINKS_PER_HOP


def test_a_hop_never_opens_more_pages_than_the_pages_limit_allows() -> None:
    setup = _setup([NOT_ENOUGH, choose(1, 2), NOT_ENOUGH], limits=AgentLimits(24, 12, 0, 3, 1))

    setup.agent.ask(QUESTION, [START])

    assert setup.links.opened == [CONTROL]


def test_the_links_of_a_followed_page_can_be_followed_in_turn() -> None:
    setup = _setup(
        [NOT_ENOUGH, choose(1), NOT_ENOUGH, choose(2), ENOUGH],
        ["The pass statement does nothing [3]."],
    )
    setup.store.pages = {START: [CONTENTS], CONTROL: [LIST_TEXT], ERRORS: [PASS_TEXT]}

    result = setup.agent.ask(QUESTION, [START])

    assert result.followed == (CONTROL, ERRORS)
    second_choice = setup.grader.requests[3][0][-1]["content"]
    assert "Errors - /tutorial/errors" in second_choice  # a link of the page that was followed
    assert "Tutorial" not in second_choice  # a page that was read is not a candidate
    assert "Control flow" not in second_choice
    assert not result.abstained


# --- when no link helps ---


def test_a_model_that_finds_no_link_worth_opening_ends_the_hops_and_the_agent_retries_the_search() -> None:
    setup = _setup([NOT_ENOUGH, choose(), REWRITTEN], limits=AgentLimits(24, 12, 1, 3, 5))

    result = setup.agent.ask(QUESTION, [START])

    assert result.abstained
    assert "No link looked worth opening." in result.steps
    assert setup.links.opened == []
    assert 'Rewrote the search as "pass statement definition".' in result.steps
    assert sum("Chose links" in step or "No link" in step for step in result.steps) == 1  # asked once, not again


def test_a_choice_that_cannot_be_read_is_taken_as_no_link() -> None:
    setup = _setup([NOT_ENOUGH, "I would open the first one"], limits=AgentLimits(24, 12, 0, 3, 5))

    result = setup.agent.ask(QUESTION, [START])

    assert result.abstained
    assert "No link looked worth opening." in result.steps
    assert setup.links.opened == []


def test_a_link_that_cannot_be_opened_is_noted_and_another_is_chosen() -> None:
    setup = _setup(
        [NOT_ENOUGH, choose(1), choose(1), ENOUGH],
        ["The pass statement does nothing [2]."],
        refused={CONTROL: "the site's robots.txt does not allow programs to open it"},
    )
    setup.store.pages = {START: [CONTENTS], CONTROL: [CONTENTS], DATA: [PASS_TEXT]}

    result = setup.agent.ask(QUESTION, [START])

    assert (
        f"Followed links: could not open {CONTROL} (the site's robots.txt does not allow programs to open it)."
        in result.steps
    )
    assert setup.links.opened == [CONTROL, DATA]  # the refused link is not offered again
    assert result.followed == (DATA,)
    assert not result.abstained


def test_when_every_link_is_refused_the_agent_goes_on_to_retry_and_then_abstains() -> None:
    refused = {CONTROL: "no", DATA: "no"}
    setup = _setup(
        [NOT_ENOUGH, choose(1, 2), REWRITTEN],
        limits=AgentLimits(24, 12, 1, 3, 5),
        refused=refused,
    )

    result = setup.agent.ask(QUESTION, [START])

    assert result.abstained
    assert result.followed == ()
    assert setup.links.opened == [CONTROL, DATA]


# --- the limits ---


def test_the_hops_are_limited() -> None:
    chain = {
        START: (Link("https://docs.example/a", "A"),),
        "https://docs.example/a": (Link("https://docs.example/b", "B"),),
    }
    setup = _setup(
        [NOT_ENOUGH, choose(1), NOT_ENOUGH],
        links=chain,
        limits=AgentLimits(24, 12, 0, 1, 5),
    )

    result = setup.agent.ask(QUESTION, [START])

    assert setup.links.opened == ["https://docs.example/a"]  # one hop, so the second link is never offered
    assert result.abstained


def test_the_pages_limit_ends_the_hops() -> None:
    setup = _setup([NOT_ENOUGH, choose(1), NOT_ENOUGH], limits=AgentLimits(24, 12, 0, 3, 1))

    setup.agent.ask(QUESTION, [START])

    assert setup.links.opened == [CONTROL]  # a second hop would need a second page


def test_no_link_is_read_or_followed_when_hops_are_switched_off() -> None:
    setup = _setup([NOT_ENOUGH], limits=AgentLimits(24, 12, 0, 0, 5))

    result = setup.agent.ask(QUESTION, [START])

    assert result.abstained
    assert setup.links.asked == []
    assert setup.links.opened == []


def test_no_link_is_followed_when_the_sources_are_not_named() -> None:
    setup = _setup([NOT_ENOUGH], limits=AgentLimits(24, 12, 0, 3, 5))

    result = setup.agent.ask(QUESTION)

    assert result.abstained
    assert setup.links.asked == []


def test_an_agent_without_a_way_to_follow_links_stays_on_its_pages() -> None:
    store = PagedStore(PAGES)
    grader = ScriptedClient(NOT_ENOUGH)
    retriever = SourceRetriever(store, FakeEmbedder(), FakeReranker())
    agent = SourceAgent(retriever, GroqChatModel(ScriptedClient()), GroqChatModel(grader), AgentLimits(24, 12, 0, 3, 5))

    assert agent.ask(QUESTION, [START]).abstained


def _graded(
    *,
    enough: bool = False,
    calls: int = 1,
    steps: int = 3,
    frontier: list[Link] | None = None,
    hops: int = 0,
    links_closed: bool = False,
    followed: list[str] | None = None,
) -> AgentState:
    return {
        "grade": Grade(sufficient=enough),
        "llm_calls": calls,
        "steps": ["step"] * steps,
        "rewrites": 0,
        "frontier": [Link(CONTROL, "Control flow")] if frontier is None else frontier,
        "hops": hops,
        "links_closed": links_closed,
        "followed": followed or [],
    }


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (_graded(), "choose"),
        (_graded(frontier=[]), "rewrite"),  # no link left to open
        (_graded(links_closed=True), "rewrite"),  # the model found none worth opening
        (_graded(hops=3), "rewrite"),  # the hops are used
        (_graded(followed=["a", "b", "c", "d", "e"]), "rewrite"),  # the pages are used
        (_graded(calls=12 - FOLLOW_CALLS + 1), "abstain"),  # a hop does not fit in the model calls, nor does a retry
        (_graded(calls=12 - FOLLOW_CALLS), "choose"),  # exactly what a hop needs
        (_graded(steps=24 - FOLLOW_STEPS + 1), "rewrite"),  # a whole hop does not fit in the steps
        (_graded(steps=24 - FOLLOW_STEPS), "choose"),
        (_graded(enough=True), "generate"),  # a verdict of enough never goes looking for more pages
    ],
)
def test_what_follows_a_grade_when_links_can_be_followed(state: AgentState, expected: str) -> None:
    assert next_step(state, LIMITS) == expected


def test_the_candidates_shown_to_the_model_are_limited() -> None:
    many = {
        START: tuple(Link(f"https://docs.example/p{number}", f"Page {number}") for number in range(MAX_CANDIDATES + 20))
    }
    setup = _setup([NOT_ENOUGH, choose(), NOT_ENOUGH], links=many, limits=AgentLimits(24, 12, 0, 3, 5))

    setup.agent.ask(QUESTION, [START])

    shown = setup.grader.requests[1][0][-1]["content"].split("Links:\n")[1].splitlines()
    assert len(shown) == MAX_CANDIDATES


def test_the_links_most_like_the_question_are_shown_when_there_are_too_many() -> None:
    boring = tuple(Link(f"https://docs.example/p{number}", f"Page {number}") for number in range(MAX_CANDIDATES + 20))
    wanted = Link("https://docs.example/tutorial/statements", "More about the pass statement")
    setup = _setup(
        [NOT_ENOUGH, choose(1), NOT_ENOUGH], links={START: (*boring, wanted)}, limits=AgentLimits(24, 12, 0, 1, 5)
    )

    setup.agent.ask(QUESTION, [START])

    assert setup.links.opened == [wanted.url]  # the link that shares words with the question is number 1


def test_the_links_in_the_same_part_of_the_site_as_the_given_page_are_listed_first() -> None:
    index = "https://docs.example/3/tutorial/index.html"
    chapter = Link("https://docs.example/3/tutorial/controlflow.html", "4. More Control Flow Tools")
    reference = Link("https://docs.example/3/reference/index.html", "The Language Reference")
    setup = _setup(
        [NOT_ENOUGH, choose(1), NOT_ENOUGH],
        links={index: (reference, chapter)},  # the page lists the reference first
        pages={index: [CONTENTS], chapter.url: [PASS_TEXT]},
        limits=AgentLimits(24, 12, 0, 1, 5),
    )

    setup.agent.ask(QUESTION, [index])

    shown = setup.grader.requests[1][0][-1]["content"].split("Links:\n")[1].splitlines()
    assert shown == [
        "1. 4. More Control Flow Tools - /3/tutorial/controlflow.html",
        "2. The Language Reference - /3/reference/index.html",
    ]
    assert setup.links.opened == [chapter.url]  # number 1 is the link near the page that was given


def test_a_page_that_was_followed_does_not_move_the_part_that_is_preferred() -> None:
    index = "https://docs.example/3/tutorial/index.html"
    library = Link("https://docs.example/3/library/os.html", "os")
    chapter = Link("https://docs.example/3/tutorial/controlflow.html", "4. More Control Flow Tools")
    reference = Link("https://docs.example/3/reference/index.html", "The Language Reference")
    statements = Link("https://docs.example/3/reference/simple_stmts.html", "7. Simple statements")
    setup = _setup(
        [NOT_ENOUGH, choose(3), NOT_ENOUGH, choose(), NOT_ENOUGH],
        links={index: (library, chapter, reference), reference.url: (statements,)},
        pages={index: [CONTENTS], reference.url: [CONTENTS]},
        limits=AgentLimits(24, 12, 0, 2, 5),
    )

    setup.agent.ask(QUESTION, [index])  # the first hop opens the reference, which is not near the page that was given

    second = setup.grader.requests[3][0][-1]["content"].split("Links:\n")[1].splitlines()
    assert [line.split(" - ")[1] for line in second] == [
        "/3/tutorial/controlflow.html",  # near the page that was given
        "/3/library/os.html",
        "/3/reference/simple_stmts.html",  # near the page that was followed, which does not count
    ]


def test_a_link_that_leads_to_a_page_already_read_adds_nothing() -> None:
    class Redirecting(FakeLinks):
        def follow(self, url: str) -> FollowResult:
            self.opened.append(url)
            return FollowResult(url=START, loaded=True)  # the link ended on the page that was given

    store = PagedStore(PAGES)
    links = Redirecting(LINKS)
    grader = ScriptedClient(NOT_ENOUGH, choose(1), NOT_ENOUGH)
    retriever = SourceRetriever(store, FakeEmbedder(), FakeReranker())
    agent = SourceAgent(
        retriever, GroqChatModel(ScriptedClient()), GroqChatModel(grader), AgentLimits(24, 12, 0, 3, 5), links
    )

    result = agent.ask(QUESTION, [START])

    assert result.followed == ()
    assert f"Followed links: {START} was already read." in result.steps
