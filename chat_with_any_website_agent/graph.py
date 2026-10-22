"""The agent graph: search a source, grade what was found, follow links or retry, then answer or abstain.

An answer has checked citations. When the source does not cover the question, the graph says so instead.
The nodes run in this order:

1. ``analyze`` tidies the question and lists the links of the pages it was given.
2. ``retrieve`` searches the pages and keeps the best excerpts. After a rewrite, a search that finds only excerpts
   that were found before ends the question in ``abstain``: the same excerpts would get the same grade.
3. ``grade`` asks a model whether the excerpts are enough. If they are, ``generate`` writes the answer. If they are
   not, the graph first looks for the answer on other pages of the site: ``choose`` shows a model a short list of the
   links not opened yet, as numbers, and ``follow`` opens the ones it chose (the site's ``robots.txt`` allowing) and
   the graph searches all the pages again. When no link is worth opening, or none can be opened any more, and a retry
   fits in the limits, ``rewrite`` makes a better query and the graph goes back to ``retrieve``. Otherwise the graph
   goes to ``abstain``.
4. ``generate`` writes the answer. If the model says that the excerpts hold no answer the graph goes to ``abstain``,
   otherwise to ``verify``.
5. ``verify`` checks the citations in plain code. If they hold, the graph ends with the answer. If they do not and
   the limits allow, ``repair`` shows the model its answer and the problem and asks it once to correct the answer,
   which is checked again. Otherwise the graph goes to ``abstain``.
6. ``abstain`` ends the graph with the message that the source does not cover the question.

Five limits keep a question from running away: the retries (``max_rewrites``), the times links are chosen
(``max_link_hops``), the pages loaded by following links (``max_pages``), the calls to the language model
(``max_llm_calls``) and the nodes run (``max_steps``). The graph looks ahead, so that it stops before a limit is hit
rather than in the middle of a path. LangGraph's own recursion limit stays as a last safeguard.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from chat_with_any_website_agent.citations import Citation, format_excerpts, verify_answer
from chat_with_any_website_agent.links import shortlist_links
from chat_with_any_website_agent.pages import describe_pages, pages_in_question
from chat_with_any_website_agent.prompts import (
    ANSWER_PROMPT,
    CHOICE_PARSER,
    CHOOSE_PROMPT,
    GRADE_PARSER,
    GRADE_PROMPT,
    NOT_COVERED_MESSAGE,
    REPAIR_PROMPT,
    REWRITE_PARSER,
    REWRITE_PROMPT,
    Grade,
    format_links,
    is_not_covered,
)
from chat_with_any_website_agent.state import AgentState, new_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel

    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.links import Link
    from chat_with_any_website_agent.navigation import FollowResult
    from chat_with_any_website_agent.pages import PageRead
    from chat_with_any_website_agent.retriever import SourceRetriever

logger = logging.getLogger(__name__)

# How many nodes the graph may still run beyond ``max_steps`` before LangGraph stops it. The look-ahead of the graph
# keeps it below ``max_steps``, so this only matters if that look-ahead is wrong.
RECURSION_MARGIN = 2

# What a way of going on needs, so that the graph can tell before it starts one whether it can finish it.
ANSWER_STEPS = 3  # generate, verify and, when the check fails, abstain
RETRY_STEPS = 6  # rewrite, retrieve, grade, and then an answer
RETRY_CALLS = 3  # rewrite, grade and generate

# Choosing links and opening them takes choose, follow, retrieve and grade, and then an answer. A hop that finds
# nothing to open is shorter, so these are what a whole hop can cost.
HOP_STEPS = 4
HOP_CALLS = 2  # choose and grade
FOLLOW_STEPS = HOP_STEPS + ANSWER_STEPS
FOLLOW_CALLS = HOP_CALLS + 1

# How many links the model is shown, and how many it may choose to open in one hop.
MAX_CANDIDATES = 40
MAX_LINKS_PER_HOP = 2

# A rejected answer is corrected once, if what is left of the limits allows it.
MAX_REPAIRS = 1
REPAIR_STEPS = 3  # repair, verify and, when the check fails again, abstain

# The reasons given when the agent does not answer.
NO_EXCERPTS_REASON = "The search found no excerpts."
NOT_ENOUGH_REASON = "The excerpts do not answer the question."
NO_ANSWER_REASON = "The model found no answer in the excerpts."
LIMIT_REASON = "A limit of steps or model calls was reached before the question could be answered."
STEP_LIMIT_REASON = "The limit of steps was reached."
UNREADABLE_GRADE_REASON = "The verdict on the excerpts could not be read."
NOTHING_NEW_NOTE = "A new search found nothing new."

type Route = Literal["generate", "choose", "rewrite", "abstain"]


@dataclass(frozen=True)
class AgentLimits:
    """The limits for one question.

    Attributes:
        max_steps: The most nodes the graph runs.
        max_llm_calls: The most calls to the language model.
        max_rewrites: The most times the search query is rewritten.
        max_link_hops: The most times the agent chooses links to open. Zero keeps it on the pages it was given.
        max_pages: The most pages loaded by following links. Zero keeps it on the pages it was given.
    """

    max_steps: int
    max_llm_calls: int
    max_rewrites: int
    max_link_hops: int = 0
    max_pages: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Take the limits from the settings."""
        return cls(
            max_steps=settings.agent_max_steps,
            max_llm_calls=settings.agent_max_llm_calls,
            max_rewrites=settings.agent_max_rewrites,
            max_link_hops=settings.agent_max_link_hops,
            max_pages=settings.agent_max_pages,
        )

    @property
    def hops(self) -> int:
        """The most hops that can happen: each one loads at least one page, so the pages limit also ends them."""
        return min(self.max_link_hops, self.max_pages)

    def problems(self) -> list[str]:
        """Say which limits are too small for the hops and retries that are allowed.

        A limit that is too small is not an error: the agent only skips the hops and retries that do not fit. But a
        copied ``.env`` can keep an old, smaller value without anybody noticing, so the notes are shown when the agent
        starts.

        Returns:
            A note for each limit that is too small to complete every hop, every retry and an answer. Empty when they
            all fit.
        """
        # read, each search and grade, each hop, each rewrite, an answer
        steps_needed = HOP_STEPS * self.hops + 3 * self.max_rewrites + 3 + ANSWER_STEPS
        calls_needed = HOP_CALLS * self.hops + 2 * self.max_rewrites + 2  # each choice, grade and rewrite, the answer
        wanted = (f"{self.hops} link hop(s), " if self.hops else "") + f"{self.max_rewrites} rewrite(s) and an answer"
        skipped = "hops and retries" if self.hops else "retries"
        notes = []
        if self.max_steps < steps_needed:
            notes.append(
                f"AGENT_MAX_STEPS is {self.max_steps}, but {wanted} can need {steps_needed} steps, "
                f"so {skipped} are skipped when they do not fit."
            )
        if self.max_llm_calls < calls_needed:
            notes.append(
                f"AGENT_MAX_LLM_CALLS is {self.max_llm_calls}, but {wanted} can need {calls_needed} calls to the "
                f"model, so {skipped} are skipped when they do not fit."
            )
        return notes


@dataclass(frozen=True)
class AgentAnswer:
    """What the agent answered to a question.

    Attributes:
        answer: The answer for the person: the checked text, or the message that the source does not cover the
            question.
        abstained: ``True`` when the agent decided not to answer from the source.
        reason: Why it did not answer, or ``None`` when it did.
        draft: The answer of the model that the check of the citations rejected, when that is why it did not answer.
        citations: The excerpts the answer cites. Empty when it did not answer.
        excerpts: The texts of the excerpts the answer is based on.
        sources: The addresses of the sources the answer cites.
        followed: The addresses of the pages the agent opened by following links, in the order it opened them.
        steps: A line for each step the agent took, in order.
        llm_calls: How many calls to the language model were made.
    """

    answer: str
    abstained: bool
    reason: str | None = None
    draft: str | None = None
    citations: tuple[Citation, ...] = ()
    excerpts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    followed: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    llm_calls: int = 0


class LinkSource(Protocol):
    """What the agent needs to follow links; ``LinkFollower`` provides it."""

    def links_of(self, url: str) -> tuple[Link, ...]:
        """The links of a stored page."""
        ...

    def follow(self, url: str) -> FollowResult:
        """Open a link and load the page behind it."""
        ...


def next_step(state: AgentState, limits: AgentLimits) -> Route:
    """Decide what follows the grading of the excerpts, within the limits.

    Args:
        state: The state after the grade.
        limits: The limits for the question.

    Returns:
        ``generate`` when the excerpts are enough and an answer fits in the limits. When they are not enough:
        ``choose`` when there are links left to open and a whole hop fits, then ``rewrite`` when a whole retry fits,
        otherwise ``abstain``.
    """
    calls_left = limits.max_llm_calls - state["llm_calls"]
    steps_used = len(state["steps"])
    if state["grade"].sufficient:
        fits = calls_left >= 1 and steps_used + ANSWER_STEPS <= limits.max_steps
        return "generate" if fits else "abstain"
    if state.get("pages"):
        return "abstain"  # the answer has to come from the named pages, so no other link or search can help
    hop = (
        bool(state.get("frontier"))
        and not state.get("links_closed", False)
        and state.get("hops", 0) < limits.max_link_hops
        and len(state.get("followed", [])) < limits.max_pages
        and calls_left >= FOLLOW_CALLS
        and steps_used + FOLLOW_STEPS <= limits.max_steps
    )
    if hop:
        return "choose"
    retry = (
        state["rewrites"] < limits.max_rewrites
        and calls_left >= RETRY_CALLS
        and steps_used + RETRY_STEPS <= limits.max_steps
    )
    return "rewrite" if retry else "abstain"


class PageSource(Protocol):
    """What the agent needs to read the pages of a PDF file that a question names; ``PageReader`` provides it."""

    def read(self, urls: Sequence[str], numbers: Sequence[int], query: str) -> PageRead:
        """Read the named pages of the PDF files among the sources."""
        ...


class SourceAgent:
    """Answers questions about stored sources, or says that they do not cover them."""

    def __init__(
        self,
        retriever: SourceRetriever,
        answer_model: BaseChatModel,
        grade_model: BaseChatModel,
        limits: AgentLimits,
        links: LinkSource | None = None,
        *,
        pages: PageSource | None = None,
    ) -> None:
        """Create the agent.

        Args:
            retriever: Finds the excerpts. Its sources are replaced by the ones of each question.
            answer_model: The model that writes the answer, for example the large Groq model.
            grade_model: The model that grades the excerpts, rewrites the query and chooses links, for example the
                small one.
            limits: The limits for one question.
            links: Reads the links of stored pages and opens them. Without it the agent stays on the pages it is given.
            pages: Reads the pages of a PDF file that the question names. Without it a page number in a question is
                only part of the question.
        """
        self._retriever = retriever
        self._limits = limits
        self._links = links
        self._pages = pages
        json_mode = {"response_format": {"type": "json_object"}, "temperature": 0}
        grader = grade_model.bind(**json_mode)
        self._grade_chain = GRADE_PROMPT | grader | GRADE_PARSER
        self._rewrite_chain = REWRITE_PROMPT | grader | REWRITE_PARSER
        self._choose_chain = CHOOSE_PROMPT | grader | CHOICE_PARSER
        writer = answer_model.bind(temperature=0)
        self._answer_chain = ANSWER_PROMPT | writer | StrOutputParser()
        self._repair_chain = REPAIR_PROMPT | writer | StrOutputParser()
        self._graph = self._build()

    def ask(self, question: str, urls: Sequence[str] | None = None) -> AgentAnswer:
        """Answer a question about stored sources.

        Args:
            question: The question.
            urls: The sources to search in, or ``None`` to search all of them.

        Returns:
            The answer with its citations, or the message that the source does not cover the question.

        Raises:
            ValueError: If the question is blank.
            ServiceBusyError: If Cohere, Weaviate or Groq keeps failing after the retries.
            BudgetExceededError: If the run's limit of calls or tokens is reached.
        """
        if not question.strip():
            raise ValueError("The question is empty.")
        state = new_state(question.strip(), None if urls is None else list(urls))
        config = {"recursion_limit": self._limits.max_steps + RECURSION_MARGIN}
        try:
            final = self._graph.invoke(state, config=config)
        except GraphRecursionError:
            logger.warning("The agent stopped at the recursion limit of the graph")
            return AgentAnswer(answer=NOT_COVERED_MESSAGE, abstained=True, reason=STEP_LIMIT_REASON)
        return _answer(final)

    def _build(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("read_pages", self._read_pages)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("choose", self._choose)
        graph.add_node("follow", self._follow)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.add_node("repair", self._repair)
        graph.add_node("abstain", self._abstain)
        graph.add_edge(START, "analyze")
        graph.add_conditional_edges("analyze", _after_analyze, {"read_pages": "read_pages", "retrieve": "retrieve"})
        graph.add_conditional_edges(
            "read_pages", _after_read_pages, {"grade": "grade", "retrieve": "retrieve", "abstain": "abstain"}
        )
        graph.add_conditional_edges("retrieve", _after_retrieve, {"grade": "grade", "abstain": "abstain"})
        graph.add_conditional_edges(
            "grade",
            self._after_grade,
            {"generate": "generate", "choose": "choose", "rewrite": "rewrite", "abstain": "abstain"},
        )
        graph.add_conditional_edges(
            "choose", self._after_choose, {"follow": "follow", "rewrite": "rewrite", "abstain": "abstain"}
        )
        graph.add_conditional_edges(
            "follow",
            self._after_follow,
            {"retrieve": "retrieve", "choose": "choose", "rewrite": "rewrite", "abstain": "abstain"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_conditional_edges("generate", _after_generate, {"verify": "verify", "abstain": "abstain"})
        graph.add_conditional_edges("verify", self._after_verify, {"repair": "repair", "abstain": "abstain", END: END})
        graph.add_conditional_edges("repair", _after_generate, {"verify": "verify", "abstain": "abstain"})
        graph.add_edge("abstain", END)
        return graph.compile()

    # The nodes. Each returns only the keys it changes, and one line for the steps.

    def _analyze(self, state: AgentState) -> AgentState:
        question = " ".join(state["question"].split())
        named = pages_in_question(question) if self._pages is not None and state["urls"] else ()
        return {
            "question": question,
            "query": question,
            "pages": list(named),
            "frontier": self._frontier(state["urls"], []),
            "steps": ["Read the question."],
        }

    def _read_pages(self, state: AgentState) -> AgentState:
        """Read the pages that the question names, instead of searching by meaning.

        The answer has to come from those pages: a search of the whole document could answer about another page. When
        the sources are not PDF files, the numbers are not pages, and the question is searched as usual.
        """
        named = describe_pages(state["pages"])
        reader = self._pages
        if reader is None:  # the question names pages only when the agent can read them, so this is not reached
            return {"pages": [], "steps": [f"The question names {named}, but this agent cannot read pages."]}
        result = reader.read(state["urls"] or [], state["pages"], state["question"])
        if not result.is_pdf:
            return {
                "pages": [],
                "steps": [f"The question names {named}, but no source is a PDF file: searched as usual."],
            }
        if not result.documents:
            problem = _page_problem(result, named)
            return {"documents": [], "page_problem": problem, "reason": problem, "steps": [f"Read {named}: {problem}"]}
        found = f"Read {named} and found {len(result.documents)} excerpt(s)"
        if result.missing:
            found += f"; no text was found for {describe_pages(result.missing)}"
        keys = [_excerpt_key(document) for document in result.documents]
        return {
            "documents": list(result.documents),
            "seen": list(dict.fromkeys(keys)),
            "steps": [found + "."],
        }

    def _frontier(self, pages: Sequence[str] | None, tried: Sequence[str]) -> list[Link]:
        """The links of the pages that were not opened yet, in the order of the pages, each address once."""
        if self._links is None or pages is None or self._limits.hops == 0:
            return []
        skip = {*pages, *tried}
        found: dict[str, Link] = {}
        for page in pages:
            for link in self._links.links_of(page):
                if link.url not in skip and link.url not in found:
                    found[link.url] = link
        return list(found.values())

    def _retrieve(self, state: AgentState) -> AgentState:
        retriever = self._retriever.model_copy(update={"urls": state["urls"]})
        documents = retriever.invoke(state["query"])
        seen = state.get("seen", [])
        fresh = [key for key in dict.fromkeys(_excerpt_key(document) for document in documents) if key not in seen]
        retry = state["rewrites"] > 0
        found = f"found {len(documents)} excerpt(s)" + (f", {len(fresh)} of them new" if retry else "")
        result: AgentState = {
            "documents": documents,
            "seen": [*seen, *fresh],
            "fresh": len(fresh),
            "steps": [f'Searched for "{state["query"]}" and {found}.'],
        }
        if retry and not fresh:
            # The same excerpts would get the same grade, so the question is not graded again.
            previous = state["grade"].reason or NOT_ENOUGH_REASON
            result["reason"] = f"{previous.rstrip('. ')}. {NOTHING_NEW_NOTE}"
        return result

    def _grade(self, state: AgentState) -> AgentState:
        documents = state["documents"]
        calls = state["llm_calls"]
        if not documents:
            grade = Grade(sufficient=False, reason=NO_EXCERPTS_REASON)  # nothing to grade, so no call is made
        else:
            calls += 1
            values = {"question": state["question"], "excerpts": format_excerpts(documents)}
            try:
                grade = self._grade_chain.invoke(values)
            except OutputParserException:
                grade = Grade(sufficient=True, reason=UNREADABLE_GRADE_REASON)  # the check of the citations follows
        verdict = "enough to answer" if grade.sufficient else f"not enough ({grade.reason or NOT_ENOUGH_REASON})"
        return {"grade": grade, "llm_calls": calls, "steps": [f"Graded the excerpts: {verdict}."]}

    def _after_grade(self, state: AgentState) -> Route:
        return next_step(state, self._limits)

    def _choose(self, state: AgentState) -> AgentState:
        """Ask the model which of the links not opened yet are likely to hold what the pages read so far lack."""
        missing = state["grade"].reason or NOT_ENOUGH_REASON
        shortlist = shortlist_links(state["frontier"], f"{state['question']} {missing}", MAX_CANDIDATES)
        room = min(MAX_LINKS_PER_HOP, self._limits.max_pages - len(state["followed"]))
        values = {
            "question": state["question"],
            "missing": missing,
            "links": format_links(shortlist),
            "limit": room,
        }
        try:
            numbers = self._choose_chain.invoke(values).links
        except OutputParserException:
            numbers = []
        chosen: list[Link] = []
        for number in numbers:  # the model gives numbers, so it can only choose among the links it was shown
            if 1 <= number <= len(shortlist) and shortlist[number - 1] not in chosen:
                chosen.append(shortlist[number - 1])
            if len(chosen) == room:
                break
        result: AgentState = {
            "hops": state["hops"] + 1,
            "llm_calls": state["llm_calls"] + 1,
            "chosen": [link.url for link in chosen],
        }
        if chosen:
            result["steps"] = ["Chose links to open: " + ", ".join(_describe_link(link) for link in chosen) + "."]
        else:
            result["links_closed"] = True  # asking again would get the same answer
            result["steps"] = ["No link looked worth opening."]
        return result

    def _after_choose(self, state: AgentState) -> Literal["follow", "rewrite", "abstain"]:
        if state["chosen"]:
            return "follow"
        # No link is left to choose, and the excerpts are not enough: only a retry can go on.
        return "rewrite" if next_step(state, self._limits) == "rewrite" else "abstain"

    def _follow(self, state: AgentState) -> AgentState:
        """Open the chosen links: each page the site allows is loaded and searched from now on."""
        if self._links is None:  # pragma: no cover - a link is only chosen when there is a way to open it
            raise AssertionError("links were chosen without a way to open them")
        pages = list(state["urls"] or [])
        followed = list(state["followed"])
        tried = list(state["tried"])
        parts = []
        loaded = 0
        for url in state["chosen"]:
            tried.append(url)
            result = self._links.follow(url)
            if not result.loaded:
                parts.append(f"could not open {url} ({result.note})")
            elif result.url in pages:
                parts.append(f"{result.url} was already read")
            else:
                pages.append(result.url)
                followed.append(result.url)
                loaded += 1
                parts.append(f"opened {result.url}")
        return {
            "urls": pages,
            "followed": followed,
            "tried": tried,
            "loaded": loaded,
            "chosen": [],
            "frontier": self._frontier(pages, tried),
            "steps": ["Followed links: " + "; ".join(parts) + "."],
        }

    def _after_follow(self, state: AgentState) -> Literal["retrieve", "choose", "rewrite", "abstain"]:
        if state["loaded"]:
            return "retrieve"
        route = next_step(state, self._limits)  # nothing new was loaded: choose again, or retry, or give up
        if route == "choose":
            return "choose"
        return "rewrite" if route == "rewrite" else "abstain"

    def _rewrite(self, state: AgentState) -> AgentState:
        values = {
            "question": state["question"],
            "query": state["query"],
            "reason": state["grade"].reason or NOT_ENOUGH_REASON,
        }
        try:
            query = self._rewrite_chain.invoke(values).query
            step = f'Rewrote the search as "{query}".'
        except OutputParserException:
            query = state["question"]
            step = "The new query could not be read, so the search will use the question again."
        return {
            "query": query,
            "rewrites": state["rewrites"] + 1,
            "llm_calls": state["llm_calls"] + 1,
            "steps": [step],
        }

    def _generate(self, state: AgentState) -> AgentState:
        values = {"question": state["question"], "excerpts": format_excerpts(state["documents"])}
        answer = self._answer_chain.invoke(values).strip()
        return _written(state, answer, "Wrote an answer.")

    def _verify(self, state: AgentState) -> AgentState:
        result = verify_answer(state["answer"], state["documents"], question=state["question"])
        if result.valid:
            return {
                "answer": result.answer,  # with the citations written as [1]
                "citations": list(result.citations),
                "steps": ["Checked the citations: they hold."],
            }
        problem = result.problem or "The citations do not hold."
        return {"reason": problem, "draft": state["answer"], "steps": [f"Checked the citations: {problem}"]}

    def _after_verify(self, state: AgentState) -> Literal["repair", "abstain", "__end__"]:
        if state.get("citations"):
            return "__end__"
        calls_left = self._limits.max_llm_calls - state["llm_calls"]
        steps_used = len(state["steps"])
        fits = calls_left >= 1 and steps_used + REPAIR_STEPS <= self._limits.max_steps
        return "repair" if state.get("repairs", 0) < MAX_REPAIRS and fits else "abstain"

    def _repair(self, state: AgentState) -> AgentState:
        values = {
            "question": state["question"],
            "excerpts": format_excerpts(state["documents"]),
            "answer": state["answer"],
            "problem": state.get("reason", ""),
        }
        answer = self._repair_chain.invoke(values).strip()
        result = _written(state, answer, "Asked the model to correct its answer.")
        result["repairs"] = state.get("repairs", 0) + 1
        return result

    def _abstain(self, state: AgentState) -> AgentState:
        reason = state.get("reason") or _grade_reason(state)
        return {
            "abstained": True,
            "answer": state.get("page_problem") or NOT_COVERED_MESSAGE,
            "citations": [],
            "reason": reason,
            "steps": [f"Decided not to answer: {reason}"],
        }


def _describe_link(link: Link) -> str:
    """A link for a line of the steps: its text and its address, or only the address when it has no text."""
    return f'"{link.text}" ({link.url})' if link.text else link.url


def _excerpt_key(document: Document) -> str:
    """What identifies an excerpt: its source and its place in the source."""
    return f"{document.metadata.get('url', '')}#{document.metadata.get('chunk_index', '')}"


def _after_analyze(state: AgentState) -> Literal["read_pages", "retrieve"]:
    """A question that names pages of a PDF file reads those pages; any other question searches."""
    return "read_pages" if state.get("pages") else "retrieve"


def _after_read_pages(state: AgentState) -> Literal["grade", "retrieve", "abstain"]:
    """Grade the pages that were read. When the sources are not PDF files the question is searched as usual."""
    if not state.get("pages"):
        return "retrieve"
    return "grade" if state["documents"] else "abstain"


def _page_problem(result: PageRead, named: str) -> str:
    """What to tell the person when the pages they named have no text: the note of the document, or that it is short."""
    base = f"This PDF file has no text for {named}."
    return f"{base} {result.note}".strip() if result.note else f"{base} The document may have fewer pages."


def _after_retrieve(state: AgentState) -> Literal["grade", "abstain"]:
    """After a rewrite, a search that found nothing new ends the question: more grading cannot change the verdict."""
    return "abstain" if state["rewrites"] > 0 and state.get("fresh", 1) == 0 else "grade"


def _written(state: AgentState, answer: str, step: str) -> AgentState:
    """What a node that wrote an answer returns: the answer, or the model's statement that it has none."""
    result: AgentState = {"answer": answer, "llm_calls": state["llm_calls"] + 1, "steps": [step]}
    if is_not_covered(answer):
        result["reason"] = NO_ANSWER_REASON
        result["draft"] = ""  # an earlier rejected answer no longer explains the abstention
        result["steps"] = ["The model found no answer in the excerpts."]
    return result


def _after_generate(state: AgentState) -> Literal["verify", "abstain"]:
    return "abstain" if is_not_covered(state["answer"]) else "verify"


def _grade_reason(state: AgentState) -> str:
    """Why the agent is not answering, when no node has said so: the grade, or else a limit."""
    grade = state.get("grade")
    if grade is not None and not grade.sufficient:
        return grade.reason or NOT_ENOUGH_REASON
    return LIMIT_REASON


def _answer(state: AgentState) -> AgentAnswer:
    abstained = state.get("abstained", False)
    citations = tuple(state.get("citations", []))
    return AgentAnswer(
        answer=state.get("answer", NOT_COVERED_MESSAGE),
        abstained=abstained,
        reason=state.get("reason") if abstained else None,
        draft=(state.get("draft") or None) if abstained else None,
        citations=citations,
        excerpts=tuple(document.page_content for document in state.get("documents", [])),
        sources=tuple(dict.fromkeys(citation.url for citation in citations)),
        followed=tuple(state.get("followed", [])),
        steps=tuple(state.get("steps", [])),
        llm_calls=state.get("llm_calls", 0),
    )
