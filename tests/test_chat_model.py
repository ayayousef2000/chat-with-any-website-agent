"""Tests of the LangChain chat model over the Groq client, with a fake client instead of the network."""

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.errors import ServiceBusyError
from chat_with_any_website_agent.groq_client import ChatReply

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeClient:
    """A stand-in for the Groq client that records its requests and can fail."""

    model = "model-x"

    def __init__(self, text: str = "The answer", error: BaseException | None = None) -> None:
        self.text = text
        self.error = error
        self.requests: list[tuple[list[Any], dict[str, Any]]] = []

    def complete(self, messages: Sequence[Any], **options: Any) -> ChatReply:
        self.requests.append((list(messages), options))
        if self.error is not None:
            raise self.error
        return ChatReply(text=self.text, prompt_tokens=12, completion_tokens=3, key_number=2)


def test_the_reply_is_a_message_with_the_tokens_and_the_key_that_answered() -> None:
    model = GroqChatModel(FakeClient("Hello"))

    reply = model.invoke("hi")

    assert isinstance(reply, AIMessage)
    assert reply.content == "Hello"
    assert reply.usage_metadata == {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
    assert reply.response_metadata["model_name"] == "model-x"
    assert reply.response_metadata["key_number"] == 2


def test_messages_are_sent_with_the_roles_groq_expects() -> None:
    client = FakeClient()
    model = GroqChatModel(client)

    model.invoke([SystemMessage("Be brief."), HumanMessage("Hi"), AIMessage("Hello"), HumanMessage("And now?")])

    assert client.requests[0][0] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "And now?"},
    ]


def test_a_plain_string_is_sent_as_a_user_message() -> None:
    client = FakeClient()

    GroqChatModel(client).invoke("Hi")

    assert client.requests[0][0] == [{"role": "user", "content": "Hi"}]


def test_bound_options_and_stop_sequences_reach_the_client() -> None:
    client = FakeClient()
    model = GroqChatModel(client).bind(temperature=0, response_format={"type": "json_object"})

    model.invoke("Hi", stop=["END"])

    assert client.requests[0][1] == {"temperature": 0, "response_format": {"type": "json_object"}, "stop": ["END"]}


def test_no_stop_option_is_sent_when_there_are_no_stop_sequences() -> None:
    client = FakeClient()

    GroqChatModel(client).invoke("Hi")

    assert client.requests[0][1] == {}


def test_a_prompt_can_be_chained_with_the_model_and_a_parser() -> None:
    client = FakeClient("Paris")
    prompt = ChatPromptTemplate.from_messages([("system", "Answer in one word."), ("human", "{question}")])
    chain = prompt | GroqChatModel(client) | StrOutputParser()

    answer = chain.invoke({"question": "Capital of France?"})

    assert answer == "Paris"
    assert client.requests[0][0] == [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Capital of France?"},
    ]


def test_a_message_that_groq_cannot_be_sent_is_refused() -> None:
    client = FakeClient()
    model = GroqChatModel(client)

    with pytest.raises(ValueError, match="'tool' message cannot be sent to Groq"):
        model.invoke([HumanMessage("Hi"), ToolMessage("result", tool_call_id="1")])

    assert client.requests == []


def test_an_error_of_the_client_reaches_the_caller() -> None:
    model = GroqChatModel(FakeClient(error=ServiceBusyError("Groq is busy.", retry_after=30.0)))

    with pytest.raises(ServiceBusyError, match="busy"):
        model.invoke("Hi")


def test_the_model_identifies_itself_for_traces() -> None:
    model = GroqChatModel(FakeClient())

    assert model._llm_type == "groq"
    assert model._identifying_params == {"model": "model-x"}
