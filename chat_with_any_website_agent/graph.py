"""The agent graph: search a source, grade what was found, retry with a better query, then answer or abstain.

An answer has checked citations. When the source does not cover the question, the graph says so instead.
The nodes run in this order:

1. ``analyze`` tidies the question.
2. ``retrieve`` searches the source and keeps the best excerpts.
3. ``grade`` asks a model whether the excerpts are enough. If they are, ``generate`` writes the answer. If they are
   not and a retry fits in the limits, ``rewrite`` makes a better query and the graph goes back to ``retrieve``.
   Otherwise the graph goes to ``abstain``.
4. ``generate`` writes the answer. If the model says that the excerpts hold no answer the graph goes to ``abstain``,
   otherwise to ``verify``.
5. ``verify`` checks the citations in plain code. If they hold, the graph ends with the answer. If they do not and
   the limits allow, ``repair`` shows the model its answer and the problem and asks it once to correct the answer,
   which is checked again. Otherwise the graph goes to ``abstain``.
6. ``abstain`` ends the graph with the message that the source does not cover the question.

Three limits keep a question from running away: the retries (``max_rewrites``), the calls to the language model
(``max_llm_calls``) and the nodes run (``max_steps``). The graph looks ahead, so that it stops before a limit is hit
rather than in the middle of a path. LangGraph's own recursion limit stays as a last safeguard.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from chat_with_any_website_agent.citations import Citation, format_excerpts, verify_answer
from chat_with_any_website_agent.prompts import (
    ANSWER_PROMPT,
    GRADE_PARSER,
    GRADE_PROMPT,
    NOT_COVERED_MARKER,
    NOT_COVERED_MESSAGE,
    REPAIR_PROMPT,
    REWRITE_PARSER,
    REWRITE_PROMPT,
    Grade,
)
from chat_with_any_website_agent.state import AgentState, new_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models.chat_models import BaseChatModel

    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.retriever import SourceRetriever

logger = logging.getLogger(__name__)

# How many nodes the graph may still run beyond ``max_steps`` before LangGraph stops it. The look-ahead of the graph
# keeps it below ``max_steps``, so this only matters if that look-ahead is wrong.
RECURSION_MARGIN = 2

# What a way of going on needs, so that the graph can tell before it starts one whether it can finish it.
ANSWER_STEPS = 3  # generate, verify and, when the check fails, abstain
RETRY_STEPS = 6  # rewrite, retrieve, grade, and then an answer
RETRY_CALLS = 3  # rewrite, grade and generate

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

type Route = Literal["generate", "rewrite", "abstain"]


@dataclass(frozen=True)
class AgentLimits:
    """The limits for one question.

    Attributes:
        max_steps: The most nodes the graph runs.
        max_llm_calls: The most calls to the language model.
        max_rewrites: The most times the search query is rewritten.
    """

    max_steps: int
    max_llm_calls: int
    max_rewrites: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Take the limits from the settings."""
        return cls(
            max_steps=settings.agent_max_steps,
            max_llm_calls=settings.agent_max_llm_calls,
            max_rewrites=settings.agent_max_rewrites,
        )


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
    steps: tuple[str, ...] = ()
    llm_calls: int = 0


def next_step(state: AgentState, limits: AgentLimits) -> Route:
    """Decide what follows the grading of the excerpts, within the limits.

    Args:
        state: The state after the grade.
        limits: The limits for the question.

    Returns:
        ``generate`` when the excerpts are enough and an answer fits in the limits, ``rewrite`` when they are not
        enough and a whole retry fits, otherwise ``abstain``.
    """
    calls_left = limits.max_llm_calls - state["llm_calls"]
    steps_used = len(state["steps"])
    if state["grade"].sufficient:
        fits = calls_left >= 1 and steps_used + ANSWER_STEPS <= limits.max_steps
        return "generate" if fits else "abstain"
    retry = (
        state["rewrites"] < limits.max_rewrites
        and calls_left >= RETRY_CALLS
        and steps_used + RETRY_STEPS <= limits.max_steps
    )
    return "rewrite" if retry else "abstain"


class SourceAgent:
    """Answers questions about stored sources, or says that they do not cover them."""

    def __init__(
        self,
        retriever: SourceRetriever,
        answer_model: BaseChatModel,
        grade_model: BaseChatModel,
        limits: AgentLimits,
    ) -> None:
        """Create the agent.

        Args:
            retriever: Finds the excerpts. Its sources are replaced by the ones of each question.
            answer_model: The model that writes the answer, for example the large Groq model.
            grade_model: The model that grades the excerpts and rewrites the query, for example the small one.
            limits: The limits for one question.
        """
        self._retriever = retriever
        self._limits = limits
        json_mode = {"response_format": {"type": "json_object"}, "temperature": 0}
        grader = grade_model.bind(**json_mode)
        self._grade_chain = GRADE_PROMPT | grader | GRADE_PARSER
        self._rewrite_chain = REWRITE_PROMPT | grader | REWRITE_PARSER
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
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.add_node("repair", self._repair)
        graph.add_node("abstain", self._abstain)
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade", self._after_grade, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"}
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
        return {"question": question, "query": question, "steps": ["Read the question."]}

    def _retrieve(self, state: AgentState) -> AgentState:
        retriever = self._retriever.model_copy(update={"urls": state["urls"]})
        documents = retriever.invoke(state["query"])
        found = f"found {len(documents)} excerpt(s)"
        return {"documents": documents, "steps": [f'Searched for "{state["query"]}" and {found}.']}

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
        result = verify_answer(state["answer"], state["documents"])
        if result.valid:
            return {"citations": list(result.citations), "steps": ["Checked the citations: they hold."]}
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
            "answer": NOT_COVERED_MESSAGE,
            "citations": [],
            "reason": reason,
            "steps": [f"Decided not to answer: {reason}"],
        }


def _written(state: AgentState, answer: str, step: str) -> AgentState:
    """What a node that wrote an answer returns: the answer, or the model's statement that it has none."""
    result: AgentState = {"answer": answer, "llm_calls": state["llm_calls"] + 1, "steps": [step]}
    if _is_not_covered(answer):
        result["reason"] = NO_ANSWER_REASON
        result["draft"] = ""  # an earlier rejected answer no longer explains the abstention
        result["steps"] = ["The model found no answer in the excerpts."]
    return result


def _after_generate(state: AgentState) -> Literal["verify", "abstain"]:
    return "abstain" if _is_not_covered(state["answer"]) else "verify"


def _is_not_covered(answer: str) -> bool:
    return answer.strip(" \t\r\n.\"'`*").upper() == NOT_COVERED_MARKER


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
        steps=tuple(state.get("steps", [])),
        llm_calls=state.get("llm_calls", 0),
    )
