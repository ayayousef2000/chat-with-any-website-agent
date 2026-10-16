import pytest

from chat_with_any_website_agent.errors import (
    AgentError,
    ExtractionError,
    FetchError,
    InvalidURLError,
    ServiceBusyError,
)


@pytest.mark.parametrize("error", [InvalidURLError, FetchError, ExtractionError])
def test_specific_errors_are_agent_errors_with_a_readable_message(error: type[AgentError]) -> None:
    raised = error("could not read the page")
    assert isinstance(raised, AgentError)
    assert str(raised) == "could not read the page"


def test_a_busy_service_error_says_how_long_to_wait() -> None:
    raised = ServiceBusyError("busy, try again", retry_after=12.5)
    assert isinstance(raised, AgentError)
    assert str(raised) == "busy, try again"
    assert raised.retry_after == 12.5
