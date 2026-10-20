"""Follow-up questions: the earlier turns of a conversation, and the rewriting of a question so that it stands alone.

The server keeps no conversation. The browser holds the last few questions and answers and sends them with each new
question, so a follow-up such as "When did he die?" can be understood. The turns are text from the browser and are
treated as untrusted, like the text of a page. Only the last few are used, clipped and without citation markers, so
that they cost few tokens.
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser

from chat_with_any_website_agent.errors import ServiceBusyError
from chat_with_any_website_agent.prompts import FOLLOW_UP_PROMPT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

MAX_TURNS = 3
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 600
MAX_STANDALONE_CHARS = 500

_QUOTES = "\"'" + "".join(chr(code) for code in (0x201C, 0x201D, 0x2018, 0x2019)) + " "  # straight and curly quotes
_CITATION = re.compile(r"\s*\[\d+\]")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Turn:
    """One question of the visitor and the answer that was given to it.

    Attributes:
        question: The question.
        answer: The answer.
    """

    question: str
    answer: str


def _clip(text: str, limit: int) -> str:
    text = _SPACES.sub(" ", text).strip()
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def compact(turns: Sequence[Turn]) -> list[Turn]:
    """Keep the last few turns, short and without citation markers.

    Args:
        turns: The turns as the browser sent them, oldest first.

    Returns:
        At most ``MAX_TURNS`` turns, each question and answer clipped to its limit. Turns that are empty after
        cleaning are dropped.
    """
    cleaned = [
        Turn(_clip(turn.question, MAX_QUESTION_CHARS), _clip(_CITATION.sub("", turn.answer), MAX_ANSWER_CHARS))
        for turn in turns[-MAX_TURNS:]
    ]
    return [turn for turn in cleaned if turn.question and turn.answer]


def usable_question(original: str, rewritten: str) -> str:
    """Take the rewritten question when it is usable, and the original one otherwise.

    A model can answer with an empty reply, several lines or a long text instead of one question. Only the first line
    is used, quotes around it are removed, and a result that is empty or too long is not trusted.

    Args:
        original: The question as it was asked.
        rewritten: What the model replied.

    Returns:
        The question to search with.
    """
    lines = [line.strip() for line in rewritten.splitlines() if line.strip()]
    candidate = lines[0].strip(_QUOTES) if lines else ""
    return candidate if candidate and len(candidate) <= MAX_STANDALONE_CHARS else original


def standalone_question(model: BaseChatModel, question: str, history: Sequence[Turn]) -> str:
    """Rewrite a follow-up question so that it makes sense without the earlier turns.

    Without earlier turns nothing is rewritten and no model is asked. A model that is limited or briefly down must not
    stop a question from being answered, so then the question is used as it was written.

    Args:
        model: The small model that rewrites the question.
        question: The latest question of the visitor.
        history: The earlier turns, oldest first. Only the last few are used.

    Returns:
        The question on its own, or the question as it was when there was nothing to rewrite or the rewrite failed.

    Raises:
        BudgetExceededError: If the run's limit of calls or tokens is reached.
    """
    turns = compact(history)
    if not turns:
        return question
    earlier = "\n\n".join(f"Question: {turn.question}\nAnswer: {turn.answer}" for turn in turns)
    chain = FOLLOW_UP_PROMPT | model.bind(temperature=0) | StrOutputParser()
    try:
        return usable_question(question, chain.invoke({"earlier": earlier, "question": question}))
    except ServiceBusyError:
        logger.warning("Rewriting a follow-up question failed because a service is busy; using it as written")
        return question
