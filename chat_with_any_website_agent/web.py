"""The command ``agent-web``: a small web server on this computer, to try the agent in a browser.

It serves a page copied from the original project (``static/``): paste an address, load it, then chat about it. The page
uses the same three calls as the original project's server. ``POST /api/ingest`` loads a page or PDF file (or loads it
again) and says its title. ``POST /api/ask`` asks the agent a question about it, with the last few turns of the
conversation so that a follow-up can be understood, and answers with the text and the excerpts it cites.
``DELETE /api/page`` deletes the stored copy. How the agent got to an answer, the pages it opened and what the question
cost stay in the terminal (``agent-ask``) and are not sent to the browser.

Like ``agent-ask``, the server uses Cohere, Weaviate and Groq, so a question is asked by hand. Started with
``agent-web`` it is for one person on one computer: it listens on the loopback address only, does one thing at a time
(the free plans have small limits) and refuses a request whose ``Host`` or ``Origin`` is not this server, so that a page
of another website that is open in the same browser cannot ask questions with the keys of the person.

A hosting service starts it with uvicorn and ``create_app_from_environment`` instead (see the ``Dockerfile``), and lists
its own name in ``WEB_ALLOWED_HOSTS``. Then every visitor is limited (``RATE_LIMIT_*``), and the stored sources are
deleted when more than ``WEB_MAX_STORED_CHUNKS`` chunks are stored, so that visitors cannot fill the free plan of
Weaviate. ``/api/health`` answers at once, to anyone, and calls no outside service, for the health check of the host.
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from chat_with_any_website_agent.config import Settings, describe_validation_error, get_settings
from chat_with_any_website_agent.errors import (
    BudgetExceededError,
    ExtractionError,
    FetchError,
    InvalidURLError,
    ServiceBusyError,
    TooManyRequestsError,
)
from chat_with_any_website_agent.fetching import normalize_url
from chat_with_any_website_agent.follow_up import Turn, standalone_question
from chat_with_any_website_agent.ratelimit import ClientLimits
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

# The page and its files, which are part of the package.
STATIC_DIR = Path(__file__).parent / "static"

# Scripts and styles are served from this origin only, so injected markup could not run or load anything.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)

# More turns than the agent uses are accepted, so that a browser that sends a few extra is not refused.
MAX_HISTORY_TURNS = 10

BUSY_MESSAGE = "The server is busy with another request. Please try again in a few seconds."
BUSY_RETRY_SECONDS = 5
HEALTH_PATH = "/api/health"
FORBIDDEN_MESSAGE = "This server answers only to pages that it serves itself."


class IngestRequest(BaseModel):
    """A request to load a page or PDF file.

    Attributes:
        url: The address of the source.
        refresh: Whether to download it again even if a copy of it is stored and the page has not changed.
    """

    url: str = Field(min_length=1, max_length=2048)
    refresh: bool = False


class IngestResponse(BaseModel):
    """A source that is ready for questions.

    Attributes:
        url: The address the source is stored under.
        title: The title of the source.
    """

    url: str
    title: str


class ForgetResponse(BaseModel):
    """The result of deleting a stored source.

    Attributes:
        deleted: Whether a stored copy existed.
    """

    deleted: bool


class HistoryTurn(BaseModel):
    """An earlier question of the conversation and the answer that was given to it.

    Attributes:
        question: The question.
        answer: The answer.
    """

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)


class AskRequest(BaseModel):
    """A question about a page or PDF file.

    Attributes:
        url: The address of the source.
        question: The question.
        history: The earlier turns of the conversation, oldest first. Only the last few are used.
    """

    url: str = Field(min_length=1, max_length=2048)
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


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


class ServerError(Exception):
    """A request that could not be done, with the status and the message for the browser.

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


class Service:
    """Loads sources, answers questions and deletes stored copies, one thing at a time."""

    def __init__(self, settings: Settings, open_runtime: Callable[[UsageCounter], AgentRuntime]) -> None:
        """Create the service.

        Args:
            settings: The settings, whose keys are never allowed into a message.
            open_runtime: Connects to the services for one request. It gets the counter of the request, and its result
                is closed when the request is done.
        """
        self._settings = settings
        self._open = open_runtime
        self._running = threading.Lock()

    def ingest(self, request: IngestRequest) -> IngestResponse:
        """Make a source ready for questions, with the page as it is now.

        Args:
            request: The source, and whether to load it again whatever is stored.

        Returns:
            The address the source is stored under and its title.

        Raises:
            ServerError: If something else is running, the source cannot be loaded, a service is busy or failed, or a
                limit was reached.
        """

        def load(runtime: AgentRuntime) -> IngestResponse:
            self._make_room(runtime)
            result = runtime.ingestor.ingest(request.url, refresh=request.refresh)
            title = result.title or runtime.store.source_title(result.url) or urlsplit(result.url).hostname
            return IngestResponse(url=result.url, title=title or result.url)

        return self._guarded(load)

    def ask(self, request: AskRequest) -> AskResponse:
        """Load the source if it is not stored and answer the question about it.

        Args:
            request: The question, its source and the earlier turns.

        Returns:
            The answer and the excerpts it cites.

        Raises:
            ServerError: If something else is running, the source cannot be loaded, a service is busy or failed, or a
                limit was reached.
        """

        def answer(runtime: AgentRuntime) -> AskResponse:
            self._make_room(runtime)
            ingested = runtime.ingestor.ingest(request.url, check=True)
            history = [Turn(turn.question, turn.answer) for turn in request.history]
            question = standalone_question(runtime.grade_model, request.question, history)
            result = runtime.agent.ask(question, [ingested.url])
            sources = [
                Source(
                    number=citation.number,
                    url=citation.url,
                    title=citation.title,
                    heading=citation.heading,
                    text=citation.text,
                )
                for citation in result.citations
            ]
            return AskResponse(answer=result.answer, sources=sources)

        return self._guarded(answer)

    def forget(self, url: str) -> ForgetResponse:
        """Delete the stored copy of a source.

        Args:
            url: The address of the source.

        Returns:
            Whether a stored copy existed.

        Raises:
            ServerError: If something else is running, the address is not valid, or a service is busy or failed.
        """

        def delete(runtime: AgentRuntime) -> ForgetResponse:
            address = normalize_url(url)
            stored = runtime.store.has_source(address)
            if stored:
                runtime.store.delete_source(address)
            return ForgetResponse(deleted=stored)

        return self._guarded(delete)

    def _guarded[T](self, work: Callable[[AgentRuntime], T]) -> T:
        """Do ``work`` with the services open, unless something else is running, and turn a failure into an error."""
        if not self._running.acquire(blocking=False):
            raise ServerError(429, BUSY_MESSAGE, BUSY_RETRY_SECONDS)
        try:
            runtime = self._open(UsageCounter())
            try:
                return work(runtime)
            finally:
                runtime.close()
        except (InvalidURLError, FetchError, ExtractionError) as error:
            raise ServerError(400, str(error)) from error
        except ServiceBusyError as error:
            raise ServerError(429, str(error), error.retry_after) from error
        except BudgetExceededError as error:
            raise ServerError(429, str(error)) from error
        except Exception as error:  # a service that cannot be reached: say what kind of failure, never a key
            logger.exception("A request failed")
            raise ServerError(503, self._without_keys(f"A service failed: {type(error).__name__}: {error}")) from error
        finally:
            self._running.release()

    def _make_room(self, runtime: AgentRuntime) -> None:
        """Delete the stored sources when more chunks are stored than the limit allows, so that the next page fits.

        The sources that visitors are still using are loaded again at their next question, so nothing breaks for
        them. With no limit (0) nothing is counted or deleted.
        """
        limit = self._settings.web_max_stored_chunks
        if limit and runtime.store.count_chunks() > limit:
            logger.warning("More than %d chunks are stored; deleting the stored sources", limit)
            runtime.store.clear()

    def _without_keys(self, text: str) -> str:
        for secret in filter(None, secret_values(self._settings)):
            text = text.replace(secret, "***")
        return text


def create_app(
    settings: Settings,
    tenant: str = DEMO_TENANT,
    *,
    extra_hosts: Sequence[str] = (),
    open_runtime: Callable[[UsageCounter], AgentRuntime] | None = None,
) -> FastAPI:
    """Create the web application.

    Args:
        settings: The settings, with the keys and the limits.
        tenant: The Weaviate tenant the sources are stored in.
        extra_hosts: Further names to answer to, besides this computer and ``WEB_ALLOWED_HOSTS``.
        open_runtime: Connects to the services for one request. The default builds the agent from the settings; tests
            give a fake.

    Returns:
        The application, which serves the page at ``/`` and its files under ``/static``, and the calls
        ``/api/ingest``, ``/api/ask``, ``/api/page`` and ``/api/health``.
    """

    def connect(usage: UsageCounter) -> AgentRuntime:
        return build_runtime(settings, usage, tenant=tenant)

    service = Service(settings, open_runtime or connect)
    limits = ClientLimits.from_settings(settings)
    names = LOOPBACK_HOSTS | settings.web_host_names | frozenset(extra_hosts)
    app = FastAPI(title="Chat With Any Website Agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    @app.middleware("http")
    async def only_from_this_page(request: Request, call_next: Callable[[Request], Any]) -> Response:
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        # The health check of a host comes to the address of the instance, not to its public name, and learns nothing.
        if request.url.path != HEALTH_PATH and (
            _host_name(host).lower() not in names or (origin is not None and urlsplit(origin).netloc != host)
        ):
            return JSONResponse({"detail": FORBIDDEN_MESSAGE}, status_code=403)
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        # The page and its files carry no version, so a browser would keep an old copy after an update. "no-cache"
        # makes it ask first; an unchanged file costs a 304.
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(ServerError)
    async def handle_server_error(request: Request, error: ServerError) -> JSONResponse:
        headers = {"Retry-After": str(max(1, round(error.retry_after)))} if error.retry_after is not None else None
        return JSONResponse({"detail": str(error)}, status_code=error.status, headers=headers)

    @app.exception_handler(TooManyRequestsError)
    async def handle_too_many_requests(request: Request, error: TooManyRequestsError) -> JSONResponse:
        headers = {"Retry-After": str(max(1, round(error.retry_after)))}
        return JSONResponse({"detail": str(error)}, status_code=429, headers=headers)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get(HEALTH_PATH)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain ``def`` endpoint runs in a worker thread, so the blocking calls to the services do not stall the server.
    # The limits of the visitor are counted first, so that a refused request reaches no service.
    @app.post("/api/ingest")
    def ingest(body: IngestRequest, request: Request) -> IngestResponse:
        limits.enforce(request, "load")
        return service.ingest(body)

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        limits.enforce(request, "ask")
        return service.ask(body)

    @app.delete("/api/page")
    def forget(request: Request, url: str = Query(min_length=1, max_length=2048)) -> ForgetResponse:
        limits.enforce(request, "load")
        return service.forget(url)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def create_app_from_environment() -> FastAPI:
    """Create the application for a hosting service, from the settings in the environment.

    The tenant is ``WEAVIATE_TENANT``. The server answers to the names in ``WEB_ALLOWED_HOSTS`` and, on Render, to the
    name of the service itself, which Render gives in ``RENDER_EXTERNAL_HOSTNAME``; a custom domain has to be listed
    in ``WEB_ALLOWED_HOSTS``. Start it with ``uvicorn --factory``, which calls this function once.

    Returns:
        The application.
    """
    settings = get_settings()
    own_name = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip().lower()
    return create_app(settings, settings.weaviate_tenant, extra_hosts=[own_name] if own_name else [])


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
