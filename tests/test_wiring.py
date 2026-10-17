"""Tests of the building of the agent from the settings, with fakes instead of the real clients."""

from typing import TYPE_CHECKING, Any

from chat_with_any_website_agent import wiring
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.graph import AgentLimits
from chat_with_any_website_agent.retrying import Retrier
from chat_with_any_website_agent.usage import UsageCounter
from chat_with_any_website_agent.wiring import build_runtime

if TYPE_CHECKING:
    import pytest


class Recorder:
    """What the fakes were created with, and what was closed."""

    def __init__(self) -> None:
        self.created: dict[str, list[tuple[Any, ...]]] = {}
        self.options: dict[str, list[dict[str, Any]]] = {}
        self.closed: list[str] = []

    def make(self, name: str, args: tuple[Any, ...], options: dict[str, Any]) -> None:
        self.created.setdefault(name, []).append(args)
        self.options.setdefault(name, []).append(options)


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    env = {
        "COHERE_API_KEY": "cohere-key",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "weaviate-key",
        "GROQ_API_KEY": "groq-key",
        **overrides,
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _fakes(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    class FakeEmbedder:
        def __init__(self, *args: Any, **options: Any) -> None:
            recorder.make("embedder", args, options)

    class FakeReranker:
        def __init__(self, *args: Any, **options: Any) -> None:
            recorder.make("reranker", args, options)

    class FakeStore:
        @classmethod
        def connect(cls, *args: Any, **options: Any) -> FakeStore:
            recorder.make("store", args, options)
            return cls()

        def close(self) -> None:
            recorder.closed.append("store")

    class FakeGroq:
        model = "model-x"

        def __init__(self, model: str) -> None:
            self.model = model

        @classmethod
        def from_settings(cls, settings: Settings, *, model: str | None = None, **options: Any) -> FakeGroq:
            recorder.make("groq", (model,), options)
            return cls(model or settings.groq_model)

        def close(self) -> None:
            recorder.closed.append(f"groq {self.model}")

    monkeypatch.setattr(wiring, "CohereEmbedder", FakeEmbedder)
    monkeypatch.setattr(wiring, "CohereReranker", FakeReranker)
    monkeypatch.setattr(wiring, "WeaviateStore", FakeStore)
    monkeypatch.setattr(wiring, "GroqClient", FakeGroq)
    return recorder


def test_the_services_are_created_from_the_settings_and_share_one_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    recorder = _fakes(monkeypatch)
    usage = UsageCounter()

    build_runtime(settings, usage)

    embedder_args = recorder.created["embedder"][0]
    assert embedder_args[1:] == (settings.cohere_embed_model, settings.cohere_embed_dimension)
    assert recorder.options["embedder"][0] == {"usage": usage}
    reranker_args = recorder.created["reranker"][0]
    assert reranker_args[1] == settings.cohere_rerank_model
    assert recorder.options["reranker"][0]["usage"] is usage
    assert recorder.options["reranker"][0]["throttle"] is not None
    assert [options["usage"] for options in recorder.options["groq"]] == [usage, usage]


def test_the_store_is_opened_on_the_tenant_of_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, WEAVIATE_TENANT="eval")
    recorder = _fakes(monkeypatch)

    build_runtime(settings, UsageCounter())

    url, _, collection, tenant = recorder.created["store"][0]
    assert (url, collection, tenant) == ("https://example.weaviate.cloud", "WebsiteChunk", "eval")
    assert recorder.options["store"][0] == {"init_timeout": 30}


def test_a_tenant_can_be_given_instead_of_the_one_in_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    recorder = _fakes(monkeypatch)

    build_runtime(settings, UsageCounter(), tenant="demo")

    assert recorder.created["store"][0][3] == "demo"


def test_the_large_model_writes_the_answer_and_the_small_one_grades(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    recorder = _fakes(monkeypatch)

    runtime = build_runtime(settings, UsageCounter())

    assert [args[0] for args in recorder.created["groq"]] == [None, "openai/gpt-oss-20b"]
    assert [chat.model for chat in runtime.chats] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_a_retrier_is_passed_on_to_both_groq_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    recorder = _fakes(monkeypatch)
    retrier = Retrier(retryable=(ValueError,))

    build_runtime(settings, UsageCounter(), retrier=retrier)

    assert [options["retrier"] for options in recorder.options["groq"]] == [retrier, retrier]


def test_the_agent_gets_the_limits_of_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AGENT_MAX_REWRITES="1")
    _fakes(monkeypatch)

    runtime = build_runtime(settings, UsageCounter())

    assert runtime.agent._limits == AgentLimits(max_steps=12, max_llm_calls=8, max_rewrites=1)


def test_closing_the_runtime_closes_the_groq_clients_and_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    recorder = _fakes(monkeypatch)
    runtime = build_runtime(settings, UsageCounter())

    runtime.close()

    assert recorder.closed == ["groq openai/gpt-oss-120b", "groq openai/gpt-oss-20b", "store"]
