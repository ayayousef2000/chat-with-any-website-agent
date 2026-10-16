"""Errors whose messages are safe to show to the end user."""


class AgentError(Exception):
    """Base error whose message is safe to show to the end user."""


class InvalidURLError(AgentError):
    """The URL is malformed, unresolvable or points to a non-public address."""


class FetchError(AgentError):
    """The page or file could not be downloaded."""


class ExtractionError(AgentError):
    """The source was downloaded but has no readable text."""


class ServiceBusyError(AgentError):
    """A service said "too many requests" or is briefly unavailable, and the retries are used up.

    Attributes:
        retry_after: How many seconds to wait before trying again.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after
