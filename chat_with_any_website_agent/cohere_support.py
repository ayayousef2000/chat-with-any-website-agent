"""What the Cohere clients have in common: which failures are worth trying again, and how to close one."""

from typing import Any

import cohere.errors as cohere_errors
import httpx

from chat_with_any_website_agent.retrying import Retrier, RetryPolicy

# A limit, a brief outage or a connection that failed or timed out. The others (a wrong key, a bad request) would
# fail the same way again.
COHERE_RETRYABLE: tuple[type[BaseException], ...] = (
    cohere_errors.TooManyRequestsError,
    cohere_errors.ServiceUnavailableError,
    cohere_errors.GatewayTimeoutError,
    cohere_errors.InternalServerError,
    httpx.TransportError,
)

# The client's own retries are switched off with this option: retrying is done once, by the Retrier, with a policy
# that suits a person who is waiting.
NO_CLIENT_RETRIES = {"max_retries": 0}


def default_retrier() -> Retrier:
    """A retrier for Cohere calls with the default policy."""
    return Retrier(retryable=COHERE_RETRYABLE)


def patient_cohere_retrier() -> Retrier:
    """A retrier for a run that may wait for a limit to lift, such as an evaluation, instead of a person who waits.

    The pauses are 10, 20, 40, 60 and 60 seconds (a little longer with the random spread), so a limit of a minute has
    time to lift. The default retrier gives up after about 8 seconds.
    """
    return Retrier(COHERE_RETRYABLE, RetryPolicy(attempts=6, base_delay=10.0, max_delay=60.0, budget=300.0))


def close_client(client: Any) -> None:
    """Close the connection of a Cohere client.

    The client has no ``close`` method, but it can be used in a ``with`` statement, and leaving that statement closes
    its connection, so that is what is done here. A client without that support is left alone.

    Args:
        client: The Cohere client.
    """
    leave = getattr(client, "__exit__", None)
    if callable(leave):
        leave(None, None, None)
