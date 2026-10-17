"""A LangChain chat model over the Groq client, so that prompts can be chained with ``|``.

The model adds nothing of its own. Key rotation, the pace of each key, retries and usage counting all stay in the
Groq client; this class only turns LangChain messages into chat messages and the reply back into a message.
"""

from typing import TYPE_CHECKING, Any, Protocol

# LangChain reads the signature of ``_generate`` at run time, so this import has to stay a real import.
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: TC002
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chat_with_any_website_agent.groq_client import ChatReply

# LangChain message types that Groq understands, and the role each one has in a chat request.
ROLES = {"system": "system", "human": "user", "ai": "assistant"}


class ChatClient(Protocol):
    """What the chat model needs from a client; ``GroqClient`` provides it."""

    @property
    def model(self) -> str:
        """The model the client asks."""
        ...

    def complete(self, messages: Sequence[Any], **options: Any) -> ChatReply:
        """Send chat messages and return the reply."""
        ...


class GroqChatModel(BaseChatModel):
    """A LangChain chat model that sends its messages through a Groq client.

    Request options bound to the model, such as ``temperature`` or ``response_format``, and the ``stop`` sequences
    of a call are passed on to the client. Tokens used and the number of the key that answered are in the metadata
    of the reply.
    """

    _client: ChatClient = PrivateAttr()

    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        """Create a chat model over a client.

        Args:
            client: The client that sends the requests, for example a ``GroqClient``.
            **kwargs: Further LangChain model settings, such as ``name`` or ``tags``.
        """
        super().__init__(**kwargs)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self._client.model}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Send the messages to the client and wrap its reply.

        Raises:
            ValueError: If a message is of a type that Groq cannot be sent, such as a tool message.
            ServiceBusyError: If every Groq key stays limited, or Groq stays unreachable, after the retries.
            BudgetExceededError: If the run's limit of calls or tokens is reached.
        """
        options = dict(kwargs)
        if stop:
            options["stop"] = stop
        reply = self._client.complete([_chat_message(message) for message in messages], **options)
        message = AIMessage(
            content=reply.text,
            usage_metadata={
                "input_tokens": reply.prompt_tokens,
                "output_tokens": reply.completion_tokens,
                "total_tokens": reply.total_tokens,
            },
            response_metadata={"model_name": self._client.model, "key_number": reply.key_number},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _chat_message(message: BaseMessage) -> dict[str, str]:
    role = ROLES.get(message.type)
    if role is None:
        raise ValueError(f"A {message.type!r} message cannot be sent to Groq; use system, human or ai messages.")
    return {"role": role, "content": str(message.text)}
