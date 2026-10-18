"""The systems that the link-following agent is compared to: a plain pipeline, and one over a crawled site.

Both retrieve excerpts and ask the answer model once. There is no grading, no retry, no check of the citations and no
following of links, so what they score is what a fixed pipeline gives. The first one searches only the pages it is
given. The second one first loads the pages of the site (breadth first, within a limit of pages and depth), as a
system that indexes a whole site would, and then searches all of them. They use the same retriever, the same answer
model and the same prompt as the agent, so a difference in the scores comes from the agent's behaviour and not from
other parts.
"""

import logging
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser

from chat_with_any_website_agent.citations import format_excerpts
from chat_with_any_website_agent.crawl import CrawlLimits, crawl_site
from chat_with_any_website_agent.evaluation import Outcome
from chat_with_any_website_agent.graph import NO_ANSWER_REASON, NO_EXCERPTS_REASON
from chat_with_any_website_agent.groq_client import patient_retrier
from chat_with_any_website_agent.ingest import SOURCE_ERRORS
from chat_with_any_website_agent.prompts import ANSWER_PROMPT, NOT_COVERED_MESSAGE, is_not_covered
from chat_with_any_website_agent.wiring import build_runtime

if TYPE_CHECKING:
    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.evaluation import Case
    from chat_with_any_website_agent.usage import UsageCounter
    from chat_with_any_website_agent.wiring import AgentRuntime

logger = logging.getLogger(__name__)


class PlainSystem:
    """Answers with one search and one call to the answer model, with no further steps.

    The earlier turns of a conversation are not used: every question is answered on its own.
    """

    def __init__(self, runtime: AgentRuntime, *, crawl: CrawlLimits | None = None) -> None:
        """Create the system over a runtime.

        Args:
            runtime: The retriever, the answer model and the ingestor, for example from ``build_runtime``.
            crawl: When given, the pages of the site of each source are loaded first, within these limits, and
                searched too.
        """
        self._runtime = runtime
        self._crawl = crawl
        self._crawled: dict[str, list[str]] = {}
        self._chain = ANSWER_PROMPT | runtime.answer_model.bind(temperature=0) | StrOutputParser()

    def prepare(self, url: str) -> None:
        """Load a source and, if the system crawls, the pages around it.

        A source that cannot be loaded is only logged here; ``ask`` reports it as the answer of each question.

        Args:
            url: The address of the source.
        """
        try:
            stored = self._runtime.ingestor.ingest(url).url
            if self._crawl is not None:
                pages = crawl_site(self._runtime.follower, stored, self._crawl)
                self._crawled[url] = pages
                logger.info("Crawled %d page(s) around a source", len(pages))
        except SOURCE_ERRORS as error:
            logger.warning("A source could not be prepared: %s", type(error).__name__)

    def ask(self, case: Case) -> Outcome:
        """Answer a question from the excerpts of the sources, and the crawled pages if the system crawls.

        Args:
            case: The question.

        Returns:
            The answer with the excerpts it is based on, or a message that a source could not be loaded.
        """
        try:
            urls = [self._runtime.ingestor.ingest(url, check=False).url for url in (case.url, *case.sources)]
        except SOURCE_ERRORS as error:
            return Outcome(answer=f"The source could not be loaded: {error}", reason=str(error))
        urls = list(dict.fromkeys([*urls, *self._crawled.get(case.url, [])]))
        documents = self._runtime.retriever.model_copy(update={"urls": urls}).invoke(case.question)
        searched = f'Searched {len(urls)} page(s) for "{case.question}" and found {len(documents)} excerpt(s).'
        if not documents:
            return Outcome(answer=NOT_COVERED_MESSAGE, reason=NO_EXCERPTS_REASON, steps=(searched,))
        excerpts = tuple(document.page_content for document in documents)
        answer = self._chain.invoke({"question": case.question, "excerpts": format_excerpts(documents)}).strip()
        if is_not_covered(answer):
            return Outcome(
                answer=NOT_COVERED_MESSAGE,
                excerpts=excerpts,
                reason=NO_ANSWER_REASON,
                steps=(searched, "The model found no answer in the excerpts."),
            )
        sources = tuple(dict.fromkeys(str(document.metadata.get("url", "")) for document in documents))
        return Outcome(answer=answer, excerpts=excerpts, sources=sources, steps=(searched, "Wrote an answer."))

    def reset(self) -> None:
        """Delete everything stored in the tenant, so that the next run loads its pages from nothing."""
        self._runtime.store.clear()

    def close(self) -> None:
        """Close the connections."""
        self._runtime.close()


def create_plain_system(settings: Settings, usage: UsageCounter) -> PlainSystem:
    """Build the plain pipeline for an evaluation run: one search over the given pages and one answer.

    Args:
        settings: The settings. The Weaviate tenant to use is ``WEAVIATE_TENANT``, so set it to ``eval`` for a run.
        usage: The counter that limits the run.

    Returns:
        The system. Its Groq calls wait for a limit to lift instead of giving up early.
    """
    runtime = build_runtime(settings, usage, retrier=patient_retrier(), follow_links=False, check_followed=False)
    return PlainSystem(runtime)


def create_crawl_system(settings: Settings, usage: UsageCounter) -> PlainSystem:
    """Build the plain pipeline over a crawled site for an evaluation run.

    Args:
        settings: The settings, with the limits of the crawl (``EVAL_CRAWL_MAX_PAGES`` and ``EVAL_CRAWL_MAX_DEPTH``).
        usage: The counter that limits the run.

    Returns:
        The system. The pages are loaded when the first question about a source is prepared, and that cost is counted.
    """
    runtime = build_runtime(settings, usage, retrier=patient_retrier(), follow_links=False, check_followed=False)
    return PlainSystem(runtime, crawl=CrawlLimits(settings.eval_crawl_max_pages, settings.eval_crawl_max_depth))
