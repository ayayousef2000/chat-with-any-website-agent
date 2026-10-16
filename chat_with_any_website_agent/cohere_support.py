"""What the Cohere clients have in common: which failures are worth trying again."""

import cohere.errors as cohere_errors
import httpx

from chat_with_any_website_agent.retrying import Retrier

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
