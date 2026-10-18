"""Building the agent from the settings: the services behind it and the objects that use them.

This is the one place where the real Cohere, Weaviate and Groq clients are created for the agent. Everything the
agent does goes through the counter that is given here, so that calls and tokens are counted and can be limited.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.embeddings import CohereEmbedder
from chat_with_any_website_agent.graph import AgentLimits, SourceAgent
from chat_with_any_website_agent.groq_client import GroqClient
from chat_with_any_website_agent.ingest import SourceIngestor
from chat_with_any_website_agent.navigation import LinkFollower
from chat_with_any_website_agent.reranking import CohereReranker
from chat_with_any_website_agent.retriever import SourceRetriever
from chat_with_any_website_agent.robots import RobotsChecker
from chat_with_any_website_agent.throttle import Throttle
from chat_with_any_website_agent.vector_store import WeaviateStore

if TYPE_CHECKING:
    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.retrying import Retrier
    from chat_with_any_website_agent.usage import UsageCounter

logger = logging.getLogger(__name__)

# The window of the rerank limit of the plan, in seconds.
RERANK_WINDOW_SECONDS = 60.0


@dataclass
class AgentRuntime:
    """An agent with what it needs to prepare sources, and what has to be closed when it is done.

    Attributes:
        agent: Answers questions about the stored sources.
        ingestor: Prepares a web page or PDF file for questions.
        store: The connection to Weaviate.
        chats: The Groq clients, which hold connections.
        cohere: The Cohere embedder and reranker, which hold connections.
    """

    agent: SourceAgent
    ingestor: SourceIngestor
    store: WeaviateStore
    chats: tuple[GroqClient, ...]
    cohere: tuple[CohereEmbedder | CohereReranker, ...] = ()

    def close(self) -> None:
        """Close the connections to Groq, Cohere and Weaviate."""
        for client in (*self.chats, *self.cohere):
            client.close()
        self.store.close()


def build_runtime(
    settings: Settings,
    usage: UsageCounter,
    *,
    tenant: str | None = None,
    retrier: Retrier | None = None,
    check_followed: bool = True,
) -> AgentRuntime:
    """Connect to the services and build the agent.

    Args:
        settings: The settings, with the keys and the limits.
        usage: Counts every call and token, and stops a run at its limits.
        tenant: The Weaviate tenant to use instead of the one in the settings, for example ``demo``.
        retrier: How to try again after a Groq limit or outage. The default suits a person who is waiting.
        check_followed: Whether to ask the website if a followed page that is already stored changed. An evaluation
            turns it off, so that its answers do not depend on when each page was asked about.

    Returns:
        The agent and its parts. Call ``close`` when done.

    Raises:
        StoreError: If the Weaviate collection exists but is not multi-tenant.
    """
    embedder = CohereEmbedder(
        settings.cohere_api_key, settings.cohere_embed_model, settings.cohere_embed_dimension, usage=usage
    )
    reranker = CohereReranker(
        settings.cohere_api_key,
        settings.cohere_rerank_model,
        throttle=Throttle(settings.cohere_rerank_requests_per_minute, RERANK_WINDOW_SECONDS),
        usage=usage,
    )
    store = WeaviateStore.connect(
        settings.weaviate_url,
        settings.weaviate_api_key,
        settings.weaviate_collection,
        tenant or settings.weaviate_tenant,
        init_timeout=settings.weaviate_init_timeout_seconds,
    )
    answerer = GroqClient.from_settings(settings, usage=usage, retrier=retrier)
    grader = GroqClient.from_settings(settings, model=settings.groq_grade_model, usage=usage, retrier=retrier)
    limits = AgentLimits.from_settings(settings)
    for note in limits.problems():
        logger.warning(note)
    ingestor = SourceIngestor.from_settings(settings, store, embedder)
    follower = LinkFollower(
        ingestor, store, RobotsChecker(timeout=settings.fetch_timeout_seconds), check=check_followed
    )
    agent = SourceAgent(
        SourceRetriever(store, embedder, reranker),
        GroqChatModel(answerer),
        GroqChatModel(grader),
        limits,
        follower,
    )
    return AgentRuntime(
        agent=agent,
        ingestor=ingestor,
        store=store,
        chats=(answerer, grader),
        cohere=(embedder, reranker),
    )
