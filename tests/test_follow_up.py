"""Tests of the follow-up questions: the turns that are kept and the question that stands on its own."""

from typing import Any

import pytest

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.errors import BudgetExceededError, ServiceBusyError
from chat_with_any_website_agent.follow_up import (
    MAX_ANSWER_CHARS,
    MAX_QUESTION_CHARS,
    MAX_STANDALONE_CHARS,
    MAX_TURNS,
    Turn,
    compact,
    standalone_question,
    usable_question,
)
from chat_with_any_website_agent.groq_client import ChatReply

HISTORY = [Turn("Who wrote it?", "Guido wrote it [1].")]


class FakeClient:
    """A model that replies with a text, or fails, and records what it was asked."""

    model = "model-x"

    def __init__(self, reply: str = "", error: BaseException | None = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: list[tuple[list[Any], dict[str, Any]]] = []

    def complete(self, messages: Any, **options: Any) -> ChatReply:
        self.requests.append((list(messages), options))
        if self.error is not None:
            raise self.error
        return ChatReply(text=self.reply, prompt_tokens=10, completion_tokens=5, key_number=1)


# --- the turns that are kept ---


def test_only_the_last_few_turns_are_kept() -> None:
    turns = [Turn(f"Q{number}?", f"A{number}.") for number in range(MAX_TURNS + 2)]

    kept = compact(turns)

    assert kept == turns[-MAX_TURNS:]


def test_citation_markers_and_extra_spaces_are_removed_from_the_answers() -> None:
    kept = compact([Turn("  Who   wrote\nit? ", "Guido [1] wrote  it[2][3].\n")])

    assert kept == [Turn("Who wrote it?", "Guido wrote it.")]


def test_a_long_question_or_answer_is_clipped_with_an_ellipsis() -> None:
    kept = compact([Turn("q" * 1000, "a" * 5000)])

    assert [len(kept[0].question), len(kept[0].answer)] == [MAX_QUESTION_CHARS, MAX_ANSWER_CHARS]
    assert kept[0].question.endswith("…")
    assert kept[0].answer.endswith("…")


def test_a_turn_that_is_empty_after_cleaning_is_dropped() -> None:
    kept = compact([Turn("   ", "An answer."), Turn("A question?", " [1] "), Turn("Q?", "A.")])

    assert kept == [Turn("Q?", "A.")]


def test_no_turns_are_kept_when_there_are_none() -> None:
    assert compact([]) == []


# --- the reply of the model ---


@pytest.mark.parametrize(
    ("rewritten", "expected"),
    [
        ("When did Guido start it?", "When did Guido start it?"),
        ('"When did Guido start it?"', "When did Guido start it?"),  # straight quotes
        ("“When did Guido start it?”", "When did Guido start it?"),  # curly quotes
        ("\n\nWhen did Guido start it?\nHe started in 1989.", "When did Guido start it?"),  # only the first line
        ("", "When did he start it?"),
        ("   \n  ", "When did he start it?"),
        ("x" * (MAX_STANDALONE_CHARS + 1), "When did he start it?"),  # a long text is not a question
        ("x" * MAX_STANDALONE_CHARS, "x" * MAX_STANDALONE_CHARS),
    ],
)
def test_the_rewritten_question_is_used_only_when_it_is_usable(rewritten: str, expected: str) -> None:
    assert usable_question("When did he start it?", rewritten) == expected


# --- the question that stands alone ---


def test_a_question_without_earlier_turns_is_not_rewritten_and_the_model_is_not_asked() -> None:
    client = FakeClient()

    assert standalone_question(GroqChatModel(client), "What does pass do?", []) == "What does pass do?"
    assert client.requests == []


def test_a_follow_up_is_rewritten_by_the_model_from_the_earlier_turns() -> None:
    client = FakeClient("When did Guido start it?")

    result = standalone_question(GroqChatModel(client), "When did he start it?", HISTORY)

    assert result == "When did Guido start it?"
    [(messages, options)] = client.requests
    assert options == {"temperature": 0}
    assert "rewrite the latest question" in messages[0]["content"]
    assert messages[-1]["content"] == (
        "Earlier in the conversation:\nQuestion: Who wrote it?\nAnswer: Guido wrote it.\n\n"
        "Latest question: When did he start it?"
    )


def test_a_reply_that_cannot_be_used_leaves_the_question_as_it_was() -> None:
    client = FakeClient("")

    assert standalone_question(GroqChatModel(client), "When did he start it?", HISTORY) == "When did he start it?"


def test_a_busy_service_leaves_the_question_as_it_was(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeClient(error=ServiceBusyError("Groq is busy right now."))

    with caplog.at_level("WARNING"):
        result = standalone_question(GroqChatModel(client), "When did he start it?", HISTORY)

    assert result == "When did he start it?"
    assert "using it as written" in caplog.text


def test_a_reached_limit_is_not_hidden() -> None:
    client = FakeClient(error=BudgetExceededError("Limit.", "groq", 10))

    with pytest.raises(BudgetExceededError):
        standalone_question(GroqChatModel(client), "When did he start it?", HISTORY)


def test_text_in_the_earlier_turns_is_not_read_as_a_template() -> None:
    client = FakeClient("What is {x}?")
    history = [Turn("What is {x}?", "A {placeholder} in braces.")]

    result = standalone_question(GroqChatModel(client), "And {y}?", history)

    assert result == "What is {x}?"
    assert "A {placeholder} in braces." in client.requests[0][0][-1]["content"]
