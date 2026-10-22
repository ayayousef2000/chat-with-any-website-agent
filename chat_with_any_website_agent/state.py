"""The state that the agent graph passes from node to node."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document

from chat_with_any_website_agent.citations import Citation
from chat_with_any_website_agent.links import Link
from chat_with_any_website_agent.prompts import Grade


class AgentState(TypedDict, total=False):
    """Everything the nodes of the graph know about one question.

    A node returns only the keys it changes. Every key but ``steps`` is replaced by the new value; ``steps`` is
    added to, so that the list ends up with every step the agent took, in order.

    Attributes:
        question: The question as it was asked.
        urls: The sources to search in, or ``None`` to search all of them. Pages the agent follows are added to it.
        pages: The pages of a PDF file that the question names. While it is not empty the answer has to come from
            those pages: nothing else is searched. It is emptied when the sources are not PDF files.
        page_problem: What to tell the person when the named pages have no text, for example that a page does not
            exist or has no text that can be selected.
        query: The text the search uses. It starts as the question and is replaced when a rewrite is made.
        rewrites: How many times the query was rewritten.
        documents: The excerpts of the latest search, best first.
        seen: The keys of every excerpt that any search has found so far.
        fresh: How many excerpts the latest search found that no earlier search had found.
        grade: The verdict on those excerpts.
        frontier: The links of the pages read so far that were not opened yet, in the order of the pages.
        hops: How many times the agent has chosen links to open.
        links_closed: ``True`` once the agent has found no link worth opening, or no link can be opened any more.
        chosen: The addresses of the links chosen in the latest hop, until they are opened.
        tried: The addresses of every link the agent has tried to open, whether or not the page could be loaded.
        followed: The addresses of the pages that were opened by following a link and loaded.
        loaded: How many pages the latest hop loaded.
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
    pages: list[int]
    page_problem: str
    query: str
    rewrites: int
    documents: list[Document]
    seen: list[str]
    fresh: int
    grade: Grade
    frontier: list[Link]
    hops: int
    links_closed: bool
    chosen: list[str]
    tried: list[str]
    followed: list[str]
    loaded: int
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
        "pages": [],
        "query": question,
        "rewrites": 0,
        "repairs": 0,
        "documents": [],
        "seen": [],
        "frontier": [],
        "hops": 0,
        "links_closed": False,
        "chosen": [],
        "tried": [],
        "followed": [],
        "loaded": 0,
        "abstained": False,
        "steps": [],
        "llm_calls": 0,
    }
