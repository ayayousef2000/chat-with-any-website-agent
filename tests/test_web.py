"""Tests of the web server, with fakes instead of the agent and the services."""

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from chat_with_any_website_agent import web
from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.citations import Citation
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import (
    BudgetExceededError,
    ExtractionError,
    FetchError,
    InvalidURLError,
    ServiceBusyError,
)
from chat_with_any_website_agent.graph import AgentAnswer
from chat_with_any_website_agent.groq_client import ChatReply
from chat_with_any_website_agent.ingest import IngestResult
from chat_with_any_website_agent.web import BUSY_MESSAGE, EXIT_BAD_SETTINGS, EXIT_OK, create_app, main

if TYPE_CHECKING:
    from chat_with_any_website_agent.usage import UsageCounter

URL = "https://docs.example/control"
NORMALIZED = f"{URL}/"
GROQ_KEY = "groq-secret-key"
CITATION = Citation(number=1, url=URL, title="Control Flow", heading="pass Statements", text="The pass statement.")
ANSWERED = AgentAnswer(
    answer="The pass statement does nothing [1].",
    abstained=False,
    citations=(CITATION,),
    followed=("https://docs.example/data",),
    steps=("Read the question.", "Wrote an answer."),
    llm_calls=2,
)
ABSTAINED = AgentAnswer(
    answer="The source does not cover this question, so there is no answer to give from it.",
    abstained=True,
    reason="The excerpts do not answer the question.",
    draft="A number the page does not state.",
)
BODY = {"url": URL, "question": "What does pass do?"}


class FakeModel:
    """The small model: it replies with the rewritten question, fails, or must not be asked at all."""

    model = "model-x"

    def __init__(self, reply: str | None = None, error: BaseException | None = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: list[Any] = []

    def complete(self, messages: Any, **options: Any) -> ChatReply:
        self.requests.append(messages)
        if self.error is not None:
            raise self.error
        if self.reply is None:
            raise AssertionError("The small model was asked, but the test expected no rewrite.")
        return ChatReply(text=self.reply, prompt_tokens=10, completion_tokens=5, key_number=1)


class FakeStore:
    """Knows whether a source is stored and what its title is, and records what was deleted."""

    def __init__(self, stored: bool = True, title: str | None = "Stored Title") -> None:
        self.stored = stored
        self.title = title
        self.deleted: list[str] = []

    def has_source(self, url: str) -> bool:
        return self.stored

    def delete_source(self, url: str) -> None:
        self.deleted.append(url)

    def source_title(self, url: str) -> str | None:
        return self.title


class FakeRuntime:
    """An agent, an ingestor and a store that record how they were used, and that can fail."""

    def __init__(
        self,
        answer: AgentAnswer = ANSWERED,
        error: BaseException | None = None,
        *,
        model: FakeModel | None = None,
        store: FakeStore | None = None,
        reused: bool = False,
    ) -> None:
        self.answer = answer
        self.error = error
        self.reused = reused
        self.closed = False
        self.ingested: list[tuple[str, bool, bool]] = []
        self.asked: list[tuple[str, list[str]]] = []
        self.model = model or FakeModel()
        self.store = store or FakeStore()
        self.grade_model = GroqChatModel(self.model)
        self.ingestor = SimpleNamespace(ingest=self._ingest)
        self.agent = SimpleNamespace(ask=self._ask)

    def _ingest(self, url: str, *, refresh: bool = False, check: bool = True) -> IngestResult:
        self.ingested.append((url, refresh, check))
        if self.error is not None:
            raise self.error
        if self.reused:
            return IngestResult(url=f"{url}/", reused=True)
        return IngestResult(url=f"{url}/", reused=False, title="Control Flow", chunk_count=4)  # normalized

    def _ask(self, question: str, urls: list[str]) -> AgentAnswer:
        self.asked.append((question, urls))
        return self.answer

    def close(self) -> None:
        self.closed = True


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name, value in {
        "COHERE_API_KEY": "cohere-secret-key",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "weaviate-secret-key",
        "GROQ_API_KEY": GROQ_KEY,
    }.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _client(
    monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime | None = None
) -> tuple[TestClient, FakeRuntime, list[UsageCounter]]:
    """A client of an application whose services are fakes, and the counters it was opened with."""
    runtime = runtime or FakeRuntime()
    opened: list[UsageCounter] = []

    def open_runtime(usage: UsageCounter) -> Any:
        opened.append(usage)
        return runtime

    app = create_app(_settings(monkeypatch), open_runtime=open_runtime)
    return TestClient(app, base_url="http://127.0.0.1:8000"), runtime, opened


# --- a question ---


def test_a_question_gets_the_answer_and_the_excerpts_it_cites(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, opened = _client(monkeypatch)

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The pass statement does nothing [1].",
        "sources": [
            {
                "number": 1,
                "url": URL,
                "title": "Control Flow",
                "heading": "pass Statements",
                "text": "The pass statement.",
            }
        ],
    }
    assert runtime.asked == [("What does pass do?", [NORMALIZED])]  # the address the source is stored under
    assert runtime.ingested == [(URL, False, True)]  # a stored copy is used after the website was asked if it changed
    assert runtime.closed
    assert len(opened) == 1


def test_how_the_agent_got_there_is_not_sent_to_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    body = client.post("/api/ask", json=BODY).json()

    assert set(body) == {"answer", "sources"}  # no steps, followed pages, reasons, drafts or cost


def test_a_question_that_the_source_does_not_cover_gets_the_message_and_no_excerpts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _client(monkeypatch, FakeRuntime(ABSTAINED))

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 200
    assert response.json() == {"answer": ABSTAINED.answer, "sources": []}


def test_text_that_is_not_ascii_is_answered_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    arabic = AgentAnswer(answer="يتم التنفيذ [1]", abstained=False)
    client, _, _ = _client(monkeypatch, FakeRuntime(arabic))

    response = client.post("/api/ask", json=BODY)

    assert response.json()["answer"] == "يتم التنفيذ [1]"


def test_a_source_of_another_page_is_named_by_its_own_address(monkeypatch: pytest.MonkeyPatch) -> None:
    followed = Citation(
        number=1, url="https://docs.example/data", title="Data", heading="Sets", text="A set has no duplicates."
    )
    answer = AgentAnswer(answer="No duplicates [1].", abstained=False, citations=(followed,))
    client, _, _ = _client(monkeypatch, FakeRuntime(answer))

    (source,) = client.post("/api/ask", json=BODY).json()["sources"]

    assert (source["url"], source["title"]) == ("https://docs.example/data", "Data")


# --- failures ---


@pytest.mark.parametrize(
    "error",
    [InvalidURLError("That address is not allowed."), FetchError("Download failed."), ExtractionError("No text.")],
)
def test_a_source_that_cannot_be_loaded_is_a_bad_request(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    client, runtime, _ = _client(monkeypatch, FakeRuntime(error=error))

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 400
    assert response.json() == {"detail": str(error)}
    assert runtime.closed


def test_a_busy_service_says_how_long_to_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ServiceBusyError("Groq is busy right now. Try again in 20 seconds.", retry_after=20.4)
    client, runtime, _ = _client(monkeypatch, FakeRuntime(error=error))

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 429
    assert response.json() == {"detail": str(error)}
    assert response.headers["retry-after"] == "20"
    assert runtime.closed


def test_a_busy_service_that_does_not_say_when_it_will_be_ready_sends_no_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _client(monkeypatch, FakeRuntime(error=ServiceBusyError("Cohere is busy right now.")))

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 429
    assert "retry-after" not in response.headers


def test_a_reached_limit_is_too_many_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, _ = _client(monkeypatch, FakeRuntime(error=BudgetExceededError("Limit.", "groq", 10)))

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 429
    assert response.json() == {"detail": "Limit."}
    assert runtime.closed


def test_an_unexpected_failure_says_what_kind_of_failure_and_never_shows_a_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client, runtime, _ = _client(monkeypatch, FakeRuntime(error=RuntimeError(f"rejected the key {GROQ_KEY}")))

    with caplog.at_level("ERROR", logger=web.__name__):
        response = client.post("/api/ask", json=BODY)

    assert response.status_code == 503
    assert response.json() == {"detail": "A service failed: RuntimeError: rejected the key ***"}
    assert runtime.closed


def test_services_that_cannot_be_opened_are_reported_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(usage: UsageCounter) -> Any:
        raise ServiceBusyError("Weaviate is busy right now. Try again in 5 seconds.", retry_after=5)

    client = TestClient(create_app(_settings(monkeypatch), open_runtime=refuse), base_url="http://127.0.0.1:8000")

    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"


def test_a_second_question_while_one_is_running_is_refused_and_costs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, opened = _client(monkeypatch)
    running = client.app.state.service._running  # type: ignore[attr-defined]
    assert running.acquire(blocking=False)

    response = client.post("/api/ask", json=BODY)
    running.release()

    assert response.status_code == 429
    assert response.json() == {"detail": BUSY_MESSAGE}
    assert opened == []
    assert not runtime.ingested


def test_the_server_can_answer_again_after_a_question_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(error=FetchError("Download failed."))
    client, _, _ = _client(monkeypatch, runtime)
    assert client.post("/api/ask", json=BODY).status_code == 400

    runtime.error = None
    response = client.post("/api/ask", json=BODY)

    assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"url": URL, "question": ""},
        {"url": "", "question": "Q?"},
        {"url": URL, "question": "q" * 2001},
        {"url": "u" * 2049, "question": "Q?"},
        {"question": "Q?"},
    ],
)
def test_a_request_that_is_not_valid_is_refused_before_anything_is_opened(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, str]
) -> None:
    client, _, opened = _client(monkeypatch)

    response = client.post("/api/ask", json=body)

    assert response.status_code == 422
    assert opened == []


# --- loading a source ---


def test_a_source_is_loaded_and_its_title_and_address_are_given(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, opened = _client(monkeypatch)

    response = client.post("/api/ingest", json={"url": URL})

    assert response.status_code == 200
    assert response.json() == {"url": NORMALIZED, "title": "Control Flow"}
    assert runtime.ingested == [(URL, False, True)]
    assert runtime.closed
    assert len(opened) == 1


def test_a_source_can_be_loaded_again_whatever_is_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, _ = _client(monkeypatch)

    client.post("/api/ingest", json={"url": URL, "refresh": True})

    assert runtime.ingested == [(URL, True, True)]


def test_the_title_of_a_stored_copy_that_was_reused_is_read_from_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch, FakeRuntime(reused=True))

    assert client.post("/api/ingest", json={"url": URL}).json() == {"url": NORMALIZED, "title": "Stored Title"}


def test_a_source_without_a_title_is_named_by_its_site(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(reused=True, store=FakeStore(title=None))
    client, _, _ = _client(monkeypatch, runtime)

    assert client.post("/api/ingest", json={"url": URL}).json()["title"] == "docs.example"


def test_a_source_that_cannot_be_loaded_is_reported_when_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, _ = _client(monkeypatch, FakeRuntime(error=FetchError("Download failed.")))

    response = client.post("/api/ingest", json={"url": URL})

    assert (response.status_code, response.json()) == (400, {"detail": "Download failed."})
    assert runtime.closed


def test_loading_waits_for_what_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, opened = _client(monkeypatch)
    running = client.app.state.service._running  # type: ignore[attr-defined]
    assert running.acquire(blocking=False)

    response = client.post("/api/ingest", json={"url": URL})
    running.release()

    assert response.status_code == 429
    assert opened == []


@pytest.mark.parametrize("body", [{"url": ""}, {"url": "u" * 2049}, {}])
def test_a_request_to_load_that_is_not_valid_is_refused(monkeypatch: pytest.MonkeyPatch, body: dict[str, str]) -> None:
    client, _, opened = _client(monkeypatch)

    assert client.post("/api/ingest", json=body).status_code == 422
    assert opened == []


# --- deleting the stored copy ---


def test_the_stored_copy_of_a_source_is_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    client, _, _ = _client(monkeypatch, runtime)

    response = client.delete("/api/page", params={"url": f"{URL}#section"})

    assert response.json() == {"deleted": True}
    assert runtime.store.deleted == [URL]  # the address the source is stored under: no fragment
    assert runtime.closed


def test_deleting_a_source_that_is_not_stored_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(store=FakeStore(stored=False))
    client, _, _ = _client(monkeypatch, runtime)

    response = client.delete("/api/page", params={"url": URL})

    assert response.json() == {"deleted": False}
    assert runtime.store.deleted == []


def test_an_address_that_is_not_valid_cannot_be_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    client, _, _ = _client(monkeypatch, runtime)

    response = client.delete("/api/page", params={"url": "ftp://docs.example/file"})

    assert response.status_code == 400
    assert runtime.store.deleted == []


@pytest.mark.parametrize("params", [{}, {"url": ""}, {"url": "u" * 2049}])
def test_a_request_to_delete_that_is_not_valid_is_refused(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, str]
) -> None:
    client, _, opened = _client(monkeypatch)

    assert client.delete("/api/page", params=params).status_code == 422
    assert opened == []


# --- follow-up questions ---


HISTORY = [{"question": "Who wrote it?", "answer": "Guido wrote it [1]."}]


def test_a_follow_up_is_rewritten_with_the_earlier_turns_before_the_agent_gets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel("When did Guido start it?")
    client, runtime, _ = _client(monkeypatch, FakeRuntime(model=model))

    response = client.post("/api/ask", json={**BODY, "question": "When did he start it?", "history": HISTORY})

    assert response.status_code == 200
    assert runtime.asked == [("When did Guido start it?", [NORMALIZED])]
    (request,) = model.requests
    text = " ".join(message["content"] for message in request)
    assert "Question: Who wrote it?\nAnswer: Guido wrote it." in text  # the citation marker is left out
    assert "Latest question: When did he start it?" in text


def test_a_first_question_is_not_rewritten_and_asks_the_small_model_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client, runtime, _ = _client(monkeypatch)  # the small model fails the test if it is asked

    client.post("/api/ask", json=BODY)

    assert runtime.asked == [("What does pass do?", [NORMALIZED])]
    assert runtime.model.requests == []


def test_a_follow_up_is_asked_as_written_when_a_service_is_busy_and_cannot_rewrite_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(error=ServiceBusyError("Groq is busy right now."))
    client, runtime, _ = _client(monkeypatch, FakeRuntime(model=model))

    response = client.post("/api/ask", json={**BODY, "history": HISTORY})

    assert response.status_code == 200
    assert runtime.asked == [("What does pass do?", [NORMALIZED])]


def test_a_reached_limit_while_rewriting_a_follow_up_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(error=BudgetExceededError("Limit.", "groq", 10))
    client, runtime, _ = _client(monkeypatch, FakeRuntime(model=model))

    response = client.post("/api/ask", json={**BODY, "history": HISTORY})

    assert response.status_code == 429
    assert runtime.asked == []
    assert runtime.closed


@pytest.mark.parametrize(
    "history",
    [
        HISTORY * 11,  # more turns than the server accepts
        [{"question": "", "answer": "A."}],
        [{"question": "Q?", "answer": ""}],
        [{"question": "q" * 2001, "answer": "A."}],
        [{"question": "Q?", "answer": "a" * 10001}],
    ],
)
def test_a_history_that_is_not_valid_is_refused(monkeypatch: pytest.MonkeyPatch, history: list[Any]) -> None:
    client, _, opened = _client(monkeypatch)

    response = client.post("/api/ask", json={**BODY, "history": history})

    assert response.status_code == 422
    assert opened == []


def test_a_history_with_the_most_turns_that_are_accepted_is_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel("A question on its own?")
    client, _, _ = _client(monkeypatch, FakeRuntime(model=model))

    response = client.post("/api/ask", json={**BODY, "history": HISTORY * 10})

    assert response.status_code == 200


# --- who may ask ---


def test_the_server_says_that_it_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    assert client.get("/api/health").json() == {"status": "ok"}


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "localhost", "[::1]:8000"])
def test_the_loopback_names_are_allowed(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    client, _, _ = _client(monkeypatch)

    assert client.get("/api/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "192.168.1.5:8000", "127.0.0.1.evil.example"])
def test_a_request_for_another_name_is_refused(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    client, _, opened = _client(monkeypatch)

    response = client.post("/api/ask", json=BODY, headers={"host": host})

    assert response.status_code == 403
    assert opened == []


def test_a_request_from_a_page_of_another_website_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, opened = _client(monkeypatch)

    response = client.post("/api/ask", json=BODY, headers={"origin": "https://evil.example"})

    assert response.status_code == 403
    assert opened == []


def test_a_request_from_another_local_port_is_refused_too(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    response = client.post("/api/ask", json=BODY, headers={"origin": "http://127.0.0.1:3000"})

    assert response.status_code == 403


def test_a_request_from_the_page_of_this_server_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    response = client.post("/api/ask", json=BODY, headers={"origin": "http://127.0.0.1:8000"})

    assert response.status_code == 200


def test_the_generated_documentation_pages_are_not_served(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]


# --- the page ---


def _page_files(client: TestClient) -> list[str]:
    """The files that the page loads, as the paths in its own text."""
    html = client.get("/").text
    return re.findall(r'(?:src|href)="(/static/[^"?]+)', html)


def test_the_page_is_served_at_the_address_of_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Chat With Any Website Agent</title>" in response.text
    for element in ("ingest-form", "ask-form", "messages", "refresh-page", "forget-page", "new-chat"):
        assert f'id="{element}"' in response.text


def test_every_file_the_page_loads_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    files = _page_files(client)

    assert sorted(files) == [
        "/static/app.js",
        "/static/favicon.svg",
        "/static/favicon.svg",
        "/static/styles.css",
        "/static/theme.js",
    ]
    types = {path: client.get(path).headers["content-type"] for path in set(files)}
    assert types["/static/app.js"].startswith(("text/javascript", "application/javascript"))
    assert types["/static/theme.js"].startswith(("text/javascript", "application/javascript"))
    assert types["/static/styles.css"].startswith("text/css")
    assert types["/static/favicon.svg"].startswith("image/svg+xml")


def test_the_page_has_no_inline_script_or_style_that_the_content_policy_would_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _client(monkeypatch)
    html = client.get("/").text

    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert "<style" not in html
    assert not re.search(r"\sstyle=", html)
    assert not re.search(r"\son[a-z]+=", html)  # no inline event handlers


def test_every_response_carries_the_security_headers_and_the_page_is_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _client(monkeypatch)

    page, script, health = client.get("/"), client.get("/static/app.js"), client.get("/api/health")

    for response in (page, script, health):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert page.headers["cache-control"] == script.headers["cache-control"] == "no-cache"
    assert "cache-control" not in health.headers


def test_the_page_is_only_served_to_this_computer(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)

    assert client.get("/", headers={"host": "evil.example"}).status_code == 403
    assert client.get("/static/app.js", headers={"host": "evil.example"}).status_code == 403


def test_the_script_only_calls_what_the_server_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)
    script = (web.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    routes = {route.path for route in client.app.routes}  # type: ignore[attr-defined]

    called = set(re.findall(r'["`](/api/[a-z]+)', script))

    assert called == {"/api/ingest", "/api/ask", "/api/page"}
    assert called <= routes


def test_the_script_reads_the_fields_that_the_server_answers_with(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch)
    script = (web.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    loaded = client.post("/api/ingest", json={"url": URL}).json()
    answered = client.post("/api/ask", json=BODY).json()

    for field in ("url", "title"):
        assert f"result.{field}" in script
        assert field in loaded
    assert "result.answer" in script
    assert "result.sources" in script
    assert set(answered) == {"answer", "sources"}
    for field in ("number", "heading", "text", "url", "title"):
        assert field in answered["sources"][0]


def test_the_page_files_have_no_carriage_returns_or_leftovers_of_the_original_project() -> None:
    for path in web.STATIC_DIR.iterdir():
        text = path.read_bytes().decode("utf-8")
        assert "\r" not in text
        assert "tests/js" not in text
        assert "module.exports" not in text


# --- the default services and the command ---


def test_the_default_services_are_built_from_the_settings_on_the_demo_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    built: list[tuple[Any, dict[str, Any]]] = []
    runtime = FakeRuntime()

    def build_runtime(settings: Settings, usage: UsageCounter, **options: Any) -> FakeRuntime:
        built.append((settings, options))
        return runtime

    monkeypatch.setattr(web, "build_runtime", build_runtime)
    client = TestClient(create_app(settings), base_url="http://127.0.0.1:8000")

    client.post("/api/ask", json=BODY)

    assert built == [(settings, {"tenant": "demo"})]  # links are followed, as the settings allow


def test_a_tenant_can_be_given_to_the_application(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    built: list[dict[str, Any]] = []

    def build_runtime(settings: Settings, usage: UsageCounter, **options: Any) -> FakeRuntime:
        built.append(options)
        return FakeRuntime()

    monkeypatch.setattr(web, "build_runtime", build_runtime)

    TestClient(create_app(settings, "mine"), base_url="http://127.0.0.1:8000").post("/api/ask", json=BODY)

    assert built == [{"tenant": "mine"}]


def test_the_command_serves_the_application_on_this_computer_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: started.append({"app": app, **options}))

    code = main(["--port", "8123", "--tenant", "mine"])

    assert code == EXIT_OK
    (call,) = started
    assert (call["host"], call["port"]) == ("127.0.0.1", 8123)
    assert call["app"].state.service is not None
    assert "http://127.0.0.1:8123" in capsys.readouterr().out


def test_the_command_does_not_listen_on_other_addresses(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])

    assert "invalid choice" in capsys.readouterr().err


def test_settings_that_are_not_valid_are_named_without_their_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("COHERE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def invalid() -> Settings:
        return Settings(_env_file=None)  # type: ignore[call-arg]

    monkeypatch.setattr(web, "get_settings", invalid)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **options: pytest.fail("the server may not start"))

    code = main([])

    assert code == EXIT_BAD_SETTINGS
    assert "The settings are not valid (COHERE_API_KEY: Field required" in capsys.readouterr().err
