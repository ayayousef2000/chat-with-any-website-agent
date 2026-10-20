"""The command ``agent-web``: a small web server on this computer, to try the agent in a browser.

``POST /api/ask`` loads a page or PDF file (a stored copy is reused), asks the agent a question about it and answers
with the text and the excerpts it cites, as the original project's server does. How the agent got there, the pages it
opened and what the question cost stay in the terminal (``agent-ask``) and are not sent to the browser.

Like ``agent-ask``, the server uses Cohere, Weaviate and Groq, so a question is asked by hand. It is for one person on
one computer: it listens on the loopback address only, answers one question at a time (the free plans have small
limits) and refuses a request whose ``Host`` or ``Origin`` is not this server, so that a page of another website that
is open in the same browser cannot ask questions with the keys of the person.
"""

import argparse
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError

from chat_with_any_website_agent.config import Settings, describe_validation_error, get_settings
from chat_with_any_website_agent.errors import (
    BudgetExceededError,
    ExtractionError,
    FetchError,
    InvalidURLError,
    ServiceBusyError,
)
from chat_with_any_website_agent.smoke import secret_values
from chat_with_any_website_agent.usage import UsageCounter
from chat_with_any_website_agent.wiring import build_runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chat_with_any_website_agent.wiring import AgentRuntime

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_SETTINGS = 2

# The tenant of the Weaviate collection that the demo uses, so that it does not mix with development or evaluation.
DEMO_TENANT = "demo"
DEFAULT_PORT = 8000

# The only names this server answers to.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

BUSY_MESSAGE = "A question is already being answered. Wait until it is done."
FORBIDDEN_MESSAGE = "This server answers only to pages that it serves itself."


class AskRequest(BaseModel):
    """A question about a page or PDF file.

    Attributes:
        url: The address of the source.
        question: The question.
    """

    url: str = Field(min_length=1, max_length=2048)
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    """An excerpt that the answer cites, with the number used in the answer text.

    Attributes:
        number: The number the answer cites it by, counting from 1.
        url: The address of the page the excerpt is from. It can be a page that the agent opened by following a link.
        title: The title of that page.
        heading: The heading of the section, or an empty string.
        text: The text of the excerpt.
    """

    number: int
    url: str
    title: str
    heading: str
    text: str


class AskResponse(BaseModel):
    """An answer with the excerpts it is based on.

    Attributes:
        answer: The answer, or the message that the source does not cover the question.
        sources: The excerpts the answer cites. Empty when the source does not cover the question.
    """

    answer: str
    sources: list[Source]


class AskError(Exception):
    """A question that could not be answered, with the status and the message for the browser.

    Attributes:
        status: The HTTP status.
        retry_after: How many seconds to wait before asking again, or ``None`` when that is not known.
    """

    def __init__(self, status: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _host_name(host: str) -> str:
    """The name in a ``Host`` header, without the port and without the brackets of an IPv6 address."""
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class QuestionService:
    """Answers one question at a time."""

    def __init__(self, settings: Settings, open_runtime: Callable[[UsageCounter], AgentRuntime]) -> None:
        """Create the service.

        Args:
            settings: The settings, whose keys are never allowed into a message.
            open_runtime: Connects to the services for one question. It gets the counter of the question, and its
                result is closed when the question is done.
        """
        self._settings = settings
        self._open = open_runtime
        self._running = threading.Lock()

    def ask(self, request: AskRequest) -> AskResponse:
        """Load the source if it is not stored and answer the question about it.

        Args:
            request: The question and its source.

        Returns:
            The answer and the excerpts it cites.

        Raises:
            AskError: If a question is already running, the source cannot be loaded, a service is busy or failed, or a
                limit was reached.
        """
        if not self._running.acquire(blocking=False):
            raise AskError(429, BUSY_MESSAGE)
        try:
            return self._answer(request)
        finally:
            self._running.release()

    def _answer(self, request: AskRequest) -> AskResponse:
        try:
            runtime = self._open(UsageCounter())
            try:
                ingested = runtime.ingestor.ingest(request.url, check=True)
                answer = runtime.agent.ask(request.question, [ingested.url])
            finally:
                runtime.close()
        except (InvalidURLError, FetchError, ExtractionError) as error:
            raise AskError(400, str(error)) from error
        except ServiceBusyError as error:
            raise AskError(429, str(error), error.retry_after) from error
        except BudgetExceededError as error:
            raise AskError(429, str(error)) from error
        except Exception as error:  # a service that cannot be reached: say what kind of failure, never a key
            logger.exception("A question failed")
            raise AskError(503, self._without_keys(f"A service failed: {type(error).__name__}: {error}")) from error
        sources = [
            Source(
                number=citation.number,
                url=citation.url,
                title=citation.title,
                heading=citation.heading,
                text=citation.text,
            )
            for citation in answer.citations
        ]
        return AskResponse(answer=answer.answer, sources=sources)

    def _without_keys(self, text: str) -> str:
        for secret in filter(None, secret_values(self._settings)):
            text = text.replace(secret, "***")
        return text


def create_app(
    settings: Settings,
    tenant: str = DEMO_TENANT,
    *,
    open_runtime: Callable[[UsageCounter], AgentRuntime] | None = None,
) -> FastAPI:
    """Create the web application.

    Args:
        settings: The settings, with the keys and the limits.
        tenant: The Weaviate tenant the sources are stored in.
        open_runtime: Connects to the services for one question. The default builds the agent from the settings; tests
            give a fake.

    Returns:
        The application, which serves ``/api/ask`` and ``/api/health``.
    """

    def connect(usage: UsageCounter) -> AgentRuntime:
        return build_runtime(settings, usage, tenant=tenant)

    service = QuestionService(settings, open_runtime or connect)
    app = FastAPI(title="Chat With Any Website Agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    @app.middleware("http")
    async def only_from_this_page(request: Request, call_next: Callable[[Request], Any]) -> Response:
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if _host_name(host) not in LOOPBACK_HOSTS or (origin is not None and urlsplit(origin).netloc != host):
            return JSONResponse({"detail": FORBIDDEN_MESSAGE}, status_code=403)
        response: Response = await call_next(request)
        return response

    @app.exception_handler(AskError)
    async def handle_ask_error(request: Request, error: AskError) -> JSONResponse:
        headers = {"Retry-After": str(max(1, round(error.retry_after)))} if error.retry_after is not None else None
        return JSONResponse({"detail": str(error)}, status_code=error.status, headers=headers)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain ``def`` endpoint runs in a worker thread, so the blocking calls to the services do not stall the server.
    @app.post("/api/ask")
    def ask(request: AskRequest) -> AskResponse:
        return service.ask(request)

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-web", description=__doc__)
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"The port to listen on (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--host",
        choices=("127.0.0.1", "localhost"),
        default="127.0.0.1",
        help="The address to listen on; only this computer is allowed",
    )
    parser.add_argument(
        "--tenant", default=DEMO_TENANT, help=f"The Weaviate tenant to use (default {DEMO_TENANT}, not WEAVIATE_TENANT)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: The arguments, without the program name. They are read from the command line when ``None``.

    Returns:
        0 when the server was stopped, 2 when the settings are not valid.
    """
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as error:
        sys.stderr.write(f"The settings are not valid ({describe_validation_error(error)}). Fill in .env first.\n")
        return EXIT_BAD_SETTINGS
    sys.stdout.write(f"Open http://{args.host}:{args.port} in a browser. Press Ctrl+C to stop.\n")
    uvicorn.run(create_app(settings, args.tenant), host=args.host, port=args.port, log_level="warning")
    return EXIT_OK
