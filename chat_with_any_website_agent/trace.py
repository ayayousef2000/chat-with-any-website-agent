"""An optional local trace of the questions the agent answered, one JSON line each.

The trace stays on this machine: nothing is sent to a service. It holds the question and what the agent did (the
steps, the pages it opened, whether it answered), not the text of the answer or of the pages. It is off unless the
setting ``AGENT_TRACE_FILE`` names a file.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat_with_any_website_agent.graph import AgentAnswer

logger = logging.getLogger(__name__)


def trace_record(question: str, answer: AgentAnswer, *, now: datetime | None = None) -> dict[str, object]:
    """Describe how one question went, as plain data that can be written as JSON.

    Args:
        question: The question as it was asked.
        answer: What the agent answered.
        now: The time to record; the current time when omitted.

    Returns:
        The record: the time (UTC), the question, whether the agent abstained and why, the steps, the pages it opened by
        following links, the sources it cited and the number of calls to the language model.
    """
    return {
        "time": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "question": question,
        "abstained": answer.abstained,
        "reason": answer.reason,
        "steps": list(answer.steps),
        "followed": list(answer.followed),
        "sources": list(answer.sources),
        "llm_calls": answer.llm_calls,
    }


def append_trace(path: str, question: str, answer: AgentAnswer) -> None:
    """Add one line to the trace file. A file that cannot be written is logged and never fails the question.

    Args:
        path: The trace file. Nothing is written when it is blank.
        question: The question as it was asked.
        answer: What the agent answered.
    """
    if not path.strip():
        return
    line = json.dumps(trace_record(question, answer), ensure_ascii=False)
    try:
        with Path(path).open("a", encoding="utf-8", newline="\n") as file:
            file.write(line + "\n")
    except OSError as error:
        logger.warning("The trace file could not be written: %s", type(error).__name__)
