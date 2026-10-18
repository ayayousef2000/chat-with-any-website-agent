"""The state that the agent graph passes from node to node."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document

from chat_with_any_website_agent.citations import Citation
from chat_with_any_website_agent.prompts import Grade


class AgentState(TypedDict, total=False):
    """Everything the nodes of the graph know about one question.

    A node returns only the keys it changes. Every key but ``steps`` is replaced by the new value; ``steps`` is
    added to, so that the list ends up with every step the agent took, in order.

    Attributes:
        question: The question as it was asked.
        urls: The sources to search in, or ``None`` to search all of them.
        query: The text the search uses. It starts as the question and is replaced when a rewrite is made.
        rewrites: How many times the query was rewritten.
        documents: The excerpts of the latest search, best first.
        seen: The keys of every excerpt that any search has found so far.
        fresh: How many excerpts the latest search found that no earlier search had found.
        grade: The verdict on those excerpts.
        answer: The text of the answer, before it was checked.
        repairs: How many times the model was asked to correct a rejected answer.
        draft: The last answer that the check of the citations rejected, kept to show what was refused.
        citations: The excerpts that the checked answer cites.
        abstained: ``True`` when the agent decided not to answer from the source.
        reason: Why the agent abstained, or what the grade found missing.
        steps: A line for each step the agent took, for the person who wants to see how it worked.
        llm_calls: How many calls to the language model were made.
    """

    question: str
    urls: list[str] | None
    query: str
    rewrites: int
    documents: list[Document]
    seen: list[str]
    fresh: int
    grade: Grade
    answer: str
    repairs: int
    draft: str
    citations: list[Citation]
    abstained: bool
    reason: str
    steps: Annotated[list[str], operator.add]
    llm_calls: int


def new_state(question: str, urls: list[str] | None = None) -> AgentState:
    """Start the state for a question.

    Args:
        question: The question.
        urls: The sources to search in, or ``None`` to search all of them.

    Returns:
        A state with the counters at zero and the query set to the question.
    """
    return {
        "question": question,
        "urls": urls,
        "query": question,
        "rewrites": 0,
        "repairs": 0,
        "documents": [],
        "seen": [],
        "abstained": False,
        "steps": [],
        "llm_calls": 0,
    }
