"""Errors whose messages are safe to show to the end user."""


class AgentError(Exception):
    """Base error whose message is safe to show to the end user."""


class InvalidURLError(AgentError):
    """The URL is malformed, unresolvable or points to a non-public address."""


class FetchError(AgentError):
    """The page or file could not be downloaded."""


class ExtractionError(AgentError):
    """The source was downloaded but has no readable text."""


class DatasetError(AgentError):
    """A question set or a results file is not in the expected format."""


class StoreError(AgentError):
    """The vector store is set up in a way that cannot be used, or refused to store something."""


class ServiceBusyError(AgentError):
    """A service said "too many requests" or is briefly unavailable, and the retries are used up.

    Attributes:
        retry_after: How many seconds to wait before trying again, or ``None`` when the service did not say.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TooManyRequestsError(AgentError):
    """A visitor used the server more often than the limits allow.

    Attributes:
        retry_after: How many seconds to wait before trying again.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceededError(AgentError):
    """A run reached the limit it was given for the calls or tokens of a service.

    Attributes:
        service: The service whose limit was reached.
        limit: The limit that was reached.
    """

    def __init__(self, message: str, service: str, limit: int) -> None:
        super().__init__(message)
        self.service = service
        self.limit = limit
