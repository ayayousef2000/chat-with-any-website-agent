"""Tests of the web server, with fakes instead of the agent and the services."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from chat_with_any_website_agent import web
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


class FakeRuntime:
    """An agent and an ingestor that record how they were used, and that can fail."""

    def __init__(self, answer: AgentAnswer = ANSWERED, error: BaseException | None = None) -> None:
        self.answer = answer
        self.error = error
        self.closed = False
        self.ingested: list[tuple[str, bool]] = []
        self.asked: list[tuple[str, list[str]]] = []
        self.ingestor = SimpleNamespace(ingest=self._ingest)
        self.agent = SimpleNamespace(ask=self._ask)

    def _ingest(self, url: str, *, refresh: bool = False, check: bool = True) -> IngestResult:
        self.ingested.append((url, check))
        if self.error is not None:
            raise self.error
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
    assert runtime.ingested == [(URL, True)]  # a stored copy is used after the website was asked if it changed
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
