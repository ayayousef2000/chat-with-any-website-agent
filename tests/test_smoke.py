"""Tests of the smoke check with fake services: no call is made to Weaviate, Cohere or Groq."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import weaviate

from chat_with_any_website_agent import groq_client, smoke
from chat_with_any_website_agent.config import Settings, get_settings
from chat_with_any_website_agent.groq_client import ChatReply
from chat_with_any_website_agent.reranking import RankedDocument
from chat_with_any_website_agent.smoke import (
    Check,
    check_cohere,
    check_groq,
    check_weaviate,
    describe_plan,
    format_report,
    main,
    run_checks,
    secret_values,
)
from chat_with_any_website_agent.usage import GROQ, UsageCounter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

REQUIRED = {
    "COHERE_API_KEY": "cohere-secret",
    "WEAVIATE_URL": "https://example.weaviate.cloud",
    "WEAVIATE_API_KEY": "weaviate-secret",
    "GROQ_API_KEY": "groq-secret-1",
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run in an empty folder, so that a real .env file is never read, and without cached settings."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in {**REQUIRED, **env}.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


# --- fake Weaviate ---


class FakeTenants:
    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        self.created: list[str] = []

    def exists(self, name: str) -> bool:
        return name in self.names

    def create(self, name: str) -> None:
        self.created.append(name)
        self.names.append(name)

    def get(self) -> dict[str, Any]:
        return dict.fromkeys(self.names, object())


class FakeCollection:
    """A collection that has the field for the age of a source, as a new one does, unless told otherwise."""

    def __init__(
        self,
        *,
        multi_tenant: bool = True,
        tenants: list[str] | None = None,
        chunks: int = 0,
        properties: list[str] | None = None,
    ) -> None:
        self.properties = (
            ["url", "loaded_at", "etag", "last_modified", "content_hash", "links"]
            if properties is None
            else list(properties)
        )
        self.added: list[Any] = []
        self.config = SimpleNamespace(
            get=lambda: SimpleNamespace(
                multi_tenancy_config=SimpleNamespace(enabled=multi_tenant),
                properties=[SimpleNamespace(name=name) for name in self.properties],
            ),
            add_property=self._add_property,
        )
        self.tenants = FakeTenants(tenants or [])
        self.chunks = chunks

    def _add_property(self, prop: Any) -> None:
        self.added.append(prop)
        self.properties.append(prop.name)

    def with_tenant(self, tenant: str) -> Any:
        count = self.chunks
        return SimpleNamespace(aggregate=SimpleNamespace(over_all=lambda **kwargs: SimpleNamespace(total_count=count)))


class FakeCollections:
    def __init__(self, existing: dict[str, FakeCollection] | None = None) -> None:
        self.existing = dict(existing or {})
        self.created: list[str] = []

    def exists(self, name: str) -> bool:
        return name in self.existing

    def use(self, name: str) -> FakeCollection:
        return self.existing[name]

    def create(self, name: str, **kwargs: Any) -> FakeCollection:
        self.created.append(name)
        self.existing[name] = FakeCollection()
        return self.existing[name]


class FakeClient:
    def __init__(self, collections: FakeCollections | None = None, *, ready: bool = True) -> None:
        self.collections = collections or FakeCollections()
        self.ready = ready
        self.closed = False

    def is_ready(self) -> bool:
        return self.ready

    def close(self) -> None:
        self.closed = True


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


# --- Weaviate ---


def test_a_cluster_that_is_not_ready_is_reported_and_nothing_else_is_tried() -> None:
    checks = check_weaviate(FakeClient(ready=False), "WebsiteChunk", "development", prepare=False)

    assert [(check.name, check.ok) for check in checks] == [("connection", False)]
    assert "waking up" in checks[0].detail


def test_a_connection_that_fails_is_reported_by_the_name_of_the_error() -> None:
    class Broken(FakeClient):
        def is_ready(self) -> bool:
            raise ConnectionError("no route to host")

    checks = check_weaviate(Broken(), "WebsiteChunk", "development", prepare=False)

    assert [(check.name, check.ok) for check in checks] == [("connection", False)]
    assert checks[0].detail == "ConnectionError: no route to host"


def test_a_missing_collection_is_fine_and_is_not_created_without_prepare() -> None:
    client = FakeClient()

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=False)

    assert all(check.ok for check in checks)
    assert "does not exist yet" in _by_name(checks)["collection"].detail
    assert client.collections.created == []


def test_prepare_creates_the_collection_and_the_three_tenants() -> None:
    client = FakeClient()

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=True)

    assert client.collections.created == ["WebsiteChunk"]
    assert client.collections.existing["WebsiteChunk"].tenants.created == ["development", "eval", "demo"]
    assert all(check.ok for check in checks)
    assert "tenants: demo, development, eval; 0 chunk(s) in development" in _by_name(checks)["collection"].detail


def test_an_existing_collection_is_read_with_its_tenants_and_chunks() -> None:
    collection = FakeCollection(tenants=["development", "demo"], chunks=7)
    client = FakeClient(FakeCollections({"WebsiteChunk": collection}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=False)

    assert _by_name(checks)["collection"].detail == (
        "WebsiteChunk is multi-tenant; tenants: demo, development; 7 chunk(s) in development"
    )
    assert collection.tenants.created == []  # nothing was written


def test_a_collection_with_all_the_fields_is_reported_as_keeping_the_version_of_each_source() -> None:
    collection = FakeCollection(tenants=["development"])
    client = FakeClient(FakeCollections({"WebsiteChunk": collection}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=True)

    assert _by_name(checks)["source versions"].detail == (
        "the collection keeps when a source was loaded, which version of the page it is and its links"
    )
    assert collection.added == []  # they were all there already


def test_prepare_adds_the_fields_to_a_collection_that_was_made_without_them() -> None:
    collection = FakeCollection(tenants=["development", "eval", "demo"], properties=["url", "text"])
    client = FakeClient(FakeCollections({"WebsiteChunk": collection}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=True)

    assert [prop.name for prop in collection.added] == ["loaded_at", "etag", "last_modified", "content_hash", "links"]
    assert _by_name(checks)["source versions"].detail == (
        "the collection keeps when a source was loaded, which version of the page it is and its links; "
        "added now: loaded_at, etag, last_modified, content_hash, links"
    )
    assert all(check.ok for check in checks)


def test_prepare_adds_only_the_fields_that_are_missing() -> None:
    collection = FakeCollection(tenants=["development", "eval", "demo"], properties=["url", "loaded_at"])
    client = FakeClient(FakeCollections({"WebsiteChunk": collection}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=True)

    assert [prop.name for prop in collection.added] == ["etag", "last_modified", "content_hash", "links"]
    assert _by_name(checks)["source versions"].detail.endswith("added now: etag, last_modified, content_hash, links")


def test_without_prepare_missing_fields_are_reported_and_nothing_is_added() -> None:
    collection = FakeCollection(tenants=["development"], properties=["url", "text"])
    client = FakeClient(FakeCollections({"WebsiteChunk": collection}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=False)

    detail = _by_name(checks)["source versions"].detail
    assert "the fields loaded_at, etag, last_modified, content_hash, links are missing" in detail
    assert "a page that changed is not noticed and no link is followed" in detail
    assert "run with --prepare" in detail
    assert _by_name(checks)["source versions"].ok  # an older collection still works, only changes are not noticed
    assert collection.added == []


def test_chunks_are_not_counted_in_a_tenant_that_does_not_exist_yet() -> None:
    client = FakeClient(FakeCollections({"WebsiteChunk": FakeCollection(tenants=["demo"])}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=False)

    assert "chunk(s)" not in _by_name(checks)["collection"].detail


def test_a_collection_that_is_not_multi_tenant_is_a_failure() -> None:
    client = FakeClient(FakeCollections({"WebsiteChunk": FakeCollection(multi_tenant=False)}))

    checks = check_weaviate(client, "WebsiteChunk", "development", prepare=False)

    assert _by_name(checks)["collection"].ok is False
    assert "not multi-tenant" in _by_name(checks)["collection"].detail


def test_an_error_while_reading_the_collection_is_a_failed_check_not_a_crash() -> None:
    class Failing(FakeCollections):
        def exists(self, name: str) -> bool:
            raise TimeoutError("slow")

    checks = check_weaviate(FakeClient(Failing()), "WebsiteChunk", "development", prepare=False)

    assert _by_name(checks)["collection"].detail == "TimeoutError: slow"


# --- Cohere ---


class FakeEmbedder:
    def __init__(self, size: int = 8, error: Exception | None = None) -> None:
        self.size = size
        self.error = error

    def embed_query(self, text: str) -> list[float]:
        if self.error:
            raise self.error
        return [0.0] * self.size


class FakeReranker:
    def __init__(self, first: int = 0, error: Exception | None = None) -> None:
        self.first = first
        self.error = error

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RankedDocument]:
        if self.error:
            raise self.error
        return [RankedDocument(self.first, 0.9), RankedDocument(1 - self.first, 0.1)]


def test_cohere_passes_when_the_vector_has_the_size_asked_for_and_the_relevant_text_comes_first() -> None:
    checks = check_cohere(FakeEmbedder(8), FakeReranker(0), 8)

    assert [(check.name, check.ok) for check in checks] == [("embedding", True), ("reranking", True)]


def test_cohere_fails_when_the_vector_has_another_size_or_the_ranking_is_wrong() -> None:
    checks = check_cohere(FakeEmbedder(4), FakeReranker(1), 8)

    assert [check.ok for check in checks] == [False, False]
    assert checks[0].detail == "one vector of size 4"


def test_a_cohere_error_is_a_failed_check_with_the_name_of_the_error() -> None:
    checks = check_cohere(FakeEmbedder(error=RuntimeError("invalid token")), FakeReranker(error=ValueError("bad")), 8)

    assert [check.detail for check in checks] == ["RuntimeError: invalid token", "ValueError: bad"]


# --- Groq ---


class FakeGroqClient:
    def __init__(self, text: str = "ok", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.options: dict[str, Any] = {}

    def complete(self, messages: Any, **options: Any) -> ChatReply:
        self.options = options
        if self.error:
            raise self.error
        return ChatReply(text=self.text, prompt_tokens=5, completion_tokens=7, key_number=1)


def test_every_key_and_the_judge_model_get_their_own_check() -> None:
    keys = [FakeGroqClient(), FakeGroqClient()]

    checks = check_groq(keys, FakeGroqClient(), "judge-model")

    assert [(check.name, check.ok) for check in checks] == [
        ("key 1", True),
        ("key 2", True),
        ("judge model judge-model", True),
    ]
    assert checks[0].detail == "replied 'ok', 12 tokens"
    assert keys[0].options["reasoning_effort"] == "low"
    assert keys[0].options["max_completion_tokens"] == 512


def test_a_key_that_fails_does_not_stop_the_other_checks() -> None:
    keys = [FakeGroqClient(error=PermissionError("invalid api key")), FakeGroqClient()]

    checks = check_groq(keys, FakeGroqClient(), "judge-model")

    assert [check.ok for check in checks] == [False, True, True]
    assert checks[0].detail == "PermissionError: invalid api key"


def test_an_empty_reply_is_a_failure() -> None:
    checks = check_groq([FakeGroqClient(text="")], FakeGroqClient(), "judge-model")

    assert (checks[0].ok, checks[0].detail) == (False, "the reply was empty")


# --- the report ---


def test_the_report_has_a_line_for_each_check_and_says_how_many_passed() -> None:
    report = format_report([Check("cohere", "embedding", True, "fine"), Check("groq", "key 1", False, "broken")])

    lines = report.splitlines()
    assert lines[0].startswith("[ok]")
    assert "embedding" in lines[0]
    assert lines[1].startswith("[FAILED]")
    assert "key 1" in lines[1]
    assert lines[-1] == "1 of 2 checks failed."
    assert format_report([Check("cohere", "embedding", True, "fine")]).endswith("All 1 checks passed.")


def test_the_details_line_up_whatever_the_length_of_the_names() -> None:
    report = format_report(
        [
            Check("groq", "key 1", True, "first detail"),
            Check("groq", "judge model openai/gpt-oss-20b", True, "second detail"),
        ]
    )

    first, second = report.splitlines()[:2]
    assert first.index("first detail") == second.index("second detail")
    assert "gpt-oss-20b  second detail" in second  # a space is always left after the name


def test_a_key_can_never_reach_the_report() -> None:
    report = format_report([Check("groq", "key 1", False, "rejected sk-secret-value")], ["sk-secret-value"])

    assert "sk-secret-value" not in report
    assert "rejected ***" in report


def test_the_secret_values_are_every_key_of_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS="groq-secret-2")

    assert secret_values(settings) == ["cohere-secret", "weaviate-secret", "groq-secret-1", "groq-secret-2"]


# --- the plan ---


def test_the_plan_counts_the_calls_and_says_nothing_is_written_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS="groq-secret-2, groq-secret-3")

    plan = describe_plan(settings, ("weaviate", "cohere", "groq"), prepare=False)

    assert "nothing is written" in plan
    assert "Cohere: 2 calls" in plan
    assert "Groq: 4 calls, one for each of 3 key(s)" in plan
    assert "openai/gpt-oss-20b" in plan


def test_the_plan_says_what_prepare_would_create_and_leaves_out_unchosen_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = describe_plan(_settings(monkeypatch), ("weaviate",), prepare=True)

    assert "create WebsiteChunk and the tenants development, eval, demo" in plan
    assert (
        "add the fields loaded_at, etag, last_modified, content_hash, links (when a source was loaded, which version "
        "of the page it is and its links) if the collection lacks them"
    ) in plan
    assert "Cohere" not in plan
    assert "Groq" not in plan


# --- building the clients ---


class FakeConfiguredEmbedder:
    def __init__(self, api_key: Any, model: str, dimension: int, **kwargs: Any) -> None:
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension


class FakeConfiguredReranker:
    def __init__(self, api_key: Any, model: str, **kwargs: Any) -> None:
        pass

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RankedDocument]:
        return [RankedDocument(0, 0.9), RankedDocument(1, 0.1)]


class FakeGroq:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        message = SimpleNamespace(content="ok")
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _fake_services(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", lambda **kwargs: client)
    monkeypatch.setattr(smoke, "CohereEmbedder", FakeConfiguredEmbedder)
    monkeypatch.setattr(smoke, "CohereReranker", FakeConfiguredReranker)
    monkeypatch.setattr(smoke, "Groq", FakeGroq)
    monkeypatch.setattr(groq_client, "Groq", FakeGroq)


def test_the_checks_are_built_from_the_settings_and_every_groq_call_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS="groq-secret-2")
    client = FakeClient(FakeCollections({"WebsiteChunk": FakeCollection(tenants=["development"], chunks=3)}))
    _fake_services(monkeypatch, client)
    usage = UsageCounter()

    checks = run_checks(settings, usage)

    assert [(check.service, check.name) for check in checks] == [
        ("weaviate", "connection"),
        ("weaviate", "collection"),
        ("weaviate", "source versions"),
        ("cohere", "embedding"),
        ("cohere", "reranking"),
        ("groq", "key 1"),
        ("groq", "key 2"),
        ("groq", "judge model openai/gpt-oss-20b"),
    ]
    assert all(check.ok for check in checks)
    assert client.closed is True
    assert usage.calls == {GROQ: 3}


def test_a_service_that_cannot_even_be_reached_is_a_failed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)

    def connect(**kwargs: Any) -> None:
        raise OSError("cluster unreachable")

    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", connect)

    checks = run_checks(settings, UsageCounter(), services=("weaviate",))

    assert [(check.name, check.ok, check.detail) for check in checks] == [
        ("connection", False, "OSError: cluster unreachable")
    ]


def test_only_the_chosen_services_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)

    def connect(**kwargs: Any) -> None:
        raise AssertionError("Weaviate must not be used")

    _fake_services(monkeypatch, FakeClient())
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", connect)

    checks = run_checks(settings, UsageCounter(), services=("cohere",))

    assert [check.service for check in checks] == ["cohere", "cohere"]


# --- the command ---


def test_settings_that_are_not_valid_are_reported_by_name_without_their_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2

    error = capsys.readouterr().err
    assert "COHERE_API_KEY" in error
    assert "Fill in .env first" in error


def test_a_blank_key_is_reported_by_name(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    for name, value in {**REQUIRED, "GROQ_API_KEY": "   "}.items():
        monkeypatch.setenv(name, value)

    assert main([]) == 2

    assert "GROQ_API_KEY must not be empty" in capsys.readouterr().err


def test_a_dry_run_shows_the_plan_and_calls_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)

    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a dry run must not call any service")

    monkeypatch.setattr(smoke, "run_checks", must_not_run)

    assert main(["--dry-run"]) == 0

    assert "Groq: 2 calls" in capsys.readouterr().out


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_the_exit_code_says_whether_every_check_passed_and_no_key_is_shown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ok: bool, code: int
) -> None:
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    checks = [Check("cohere", "embedding", ok, "cohere-secret")]
    monkeypatch.setattr(smoke, "run_checks", lambda *args, **kwargs: checks)

    assert main(["--only", "cohere"]) == code

    output = capsys.readouterr().out
    assert "cohere-secret" not in output
    assert "Calls counted:" in output
