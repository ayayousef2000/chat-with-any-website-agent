"""The agent as a system for the evaluation: it prepares the sources of a question and answers it.

A source that cannot be loaded does not stop a run: it is recorded as a failed answer, so that the rest of the set is
still answered. A busy service or a reached limit does stop the run, because the harness handles those.
"""

import logging
from typing import TYPE_CHECKING

from chat_with_any_website_agent.errors import ExtractionError, FetchError, InvalidURLError
from chat_with_any_website_agent.evaluation import Outcome
from chat_with_any_website_agent.groq_client import patient_retrier
from chat_with_any_website_agent.wiring import build_runtime

if TYPE_CHECKING:
    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.evaluation import Case
    from chat_with_any_website_agent.usage import UsageCounter
    from chat_with_any_website_agent.wiring import AgentRuntime

logger = logging.getLogger(__name__)

# The ways a single source can fail without anything being wrong with the services.
SOURCE_ERRORS = (InvalidURLError, FetchError, ExtractionError)


class AgentSystem:
    """Answers the questions of an evaluation with the agent.

    The earlier turns of a conversation are not used yet: every question is answered on its own.
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        """Create the system over a runtime.

        Args:
            runtime: The agent and its ingestor, for example from ``build_runtime``.
        """
        self._runtime = runtime

    def prepare(self, url: str) -> None:
        """Load a source so that questions about it can be answered.

        A source that cannot be loaded is only logged here; ``ask`` reports it as the answer of each question.

        Args:
            url: The address of the source.
        """
        try:
            self._runtime.ingestor.ingest(url)
        except SOURCE_ERRORS as error:
            logger.warning("A source could not be prepared: %s", type(error).__name__)

    def ask(self, case: Case) -> Outcome:
        """Answer a question about its source and the further sources it names.

        Args:
            case: The question.

        Returns:
            The answer with the excerpts it is based on, and how the agent got to it, or a message that a source
            could not be loaded.
        """
        try:
            urls = [self._runtime.ingestor.ingest(url).url for url in (case.url, *case.sources)]
        except SOURCE_ERRORS as error:
            return Outcome(answer=f"The source could not be loaded: {error}", reason=str(error))
        answer = self._runtime.agent.ask(case.question, urls)
        return Outcome(
            answer=answer.answer,
            excerpts=answer.excerpts,
            sources=answer.sources,
            reason=answer.reason,
            steps=answer.steps,
            draft=answer.draft,
        )

    def close(self) -> None:
        """Close the connections of the agent."""
        self._runtime.close()


def create_agent_system(settings: Settings, usage: UsageCounter) -> AgentSystem:
    """Build the agent system for an evaluation run.

    Args:
        settings: The settings. The Weaviate tenant to use is ``WEAVIATE_TENANT``, so set it to ``eval`` for a run.
        usage: The counter that limits the run.

    Returns:
        The system. Its Groq calls wait for a limit to lift instead of giving up early.
    """
    return AgentSystem(build_runtime(settings, usage, retrier=patient_retrier()))
