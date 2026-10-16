from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from chat_with_any_website_agent.config import EMBED_DIMENSIONS, Settings, get_settings

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED = {
    "COHERE_API_KEY": "cohere-key",
    "WEAVIATE_URL": "https://example.weaviate.cloud",
    "WEAVIATE_API_KEY": "weaviate-key",
    "GROQ_API_KEY": "groq-key",
}


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    for key, value in {**REQUIRED, **overrides}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _groq_keys(settings: Settings) -> list[str]:
    return [key.get_secret_value() for key in settings.groq_keys]


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert settings.cohere_embed_model == "embed-v5.0-pro"
    assert settings.cohere_embed_dimension == 1024
    assert settings.cohere_rerank_model == "rerank-v4.0-fast"
    assert settings.weaviate_collection == "WebsiteChunk"
    assert settings.weaviate_tenant == "development"
    assert settings.weaviate_init_timeout_seconds == 30
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.groq_tokens_per_minute == 8000


def test_default_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert settings.agent_max_steps == 8
    assert settings.agent_max_link_hops == 3
    assert settings.agent_max_llm_calls == 8
    assert settings.agent_max_rewrites == 2
    assert settings.eval_max_cohere_calls == 300
    assert settings.eval_max_groq_calls == 200
    assert settings.eval_max_groq_tokens == 120_000


def test_overrides_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch, AGENT_MAX_STEPS="5", EVAL_MAX_GROQ_TOKENS="50000", GROQ_MODEL="openai/gpt-oss-20b"
    )
    assert settings.agent_max_steps == 5
    assert settings.eval_max_groq_tokens == 50_000
    assert settings.groq_model == "openai/gpt-oss-20b"


def test_missing_required_values_fail() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("name", ["COHERE_API_KEY", "WEAVIATE_API_KEY", "GROQ_API_KEY"])
def test_blank_keys_fail_with_the_name_of_the_variable(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValidationError, match=name):
        _settings(monkeypatch, **{name: "   "})


def test_keys_do_not_appear_in_the_text_form_of_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS="backup-key")
    shown = f"{settings!r} {settings!s}"
    for secret in ("cohere-key", "weaviate-key", "groq-key", "backup-key"):
        assert secret not in shown


# --- values that must be positive ---


@pytest.mark.parametrize(
    "name",
    [
        "WEAVIATE_INIT_TIMEOUT_SECONDS",
        "GROQ_TOKENS_PER_MINUTE",
        "AGENT_MAX_STEPS",
        "AGENT_MAX_LLM_CALLS",
        "EVAL_MAX_COHERE_CALLS",
        "EVAL_MAX_GROQ_CALLS",
        "EVAL_MAX_GROQ_TOKENS",
    ],
)
def test_values_that_must_be_positive_reject_zero(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch, **{name: "0"})


@pytest.mark.parametrize("name", ["AGENT_MAX_LINK_HOPS", "AGENT_MAX_REWRITES"])
def test_hops_and_rewrites_may_be_zero_but_not_negative(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    assert _settings(monkeypatch, **{name: "0"}) is not None
    with pytest.raises(ValidationError):
        _settings(monkeypatch, **{name: "-1"})


# --- the embedding size ---


@pytest.mark.parametrize("dimension", EMBED_DIMENSIONS)
def test_supported_embedding_sizes_are_accepted(monkeypatch: pytest.MonkeyPatch, dimension: int) -> None:
    assert _settings(monkeypatch, COHERE_EMBED_DIMENSION=str(dimension)).cohere_embed_dimension == dimension


@pytest.mark.parametrize("dimension", ["0", "100", "1000", "4096"])
def test_unsupported_embedding_sizes_are_rejected(monkeypatch: pytest.MonkeyPatch, dimension: str) -> None:
    with pytest.raises(ValidationError, match="cohere_embed_dimension"):
        _settings(monkeypatch, COHERE_EMBED_DIMENSION=dimension)


# --- the Weaviate cluster and tenant ---


def test_the_cluster_address_loses_spaces_and_a_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, WEAVIATE_URL=" https://example.weaviate.cloud/ ")
    assert settings.weaviate_url == "https://example.weaviate.cloud"


def test_the_cluster_address_may_be_a_host_name(monkeypatch: pytest.MonkeyPatch) -> None:
    host = "abc.c0.example.weaviate.cloud"
    assert _settings(monkeypatch, WEAVIATE_URL=host).weaviate_url == host


def test_a_blank_cluster_address_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="weaviate_url"):
        _settings(monkeypatch, WEAVIATE_URL="  ")


@pytest.mark.parametrize("name", ["development", "eval", "demo", "dev_1", "team-A", "a" * 64, "abcd"])
def test_valid_tenant_names_are_accepted(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    assert _settings(monkeypatch, WEAVIATE_TENANT=name).weaviate_tenant == name


@pytest.mark.parametrize("name", ["", "dev", "a" * 65, "has space", "slash/name", "dot.name", "accént"])
def test_invalid_tenant_names_are_rejected(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValidationError, match="weaviate_tenant"):
        _settings(monkeypatch, WEAVIATE_TENANT=name)


# --- the Groq keys ---


def test_there_are_no_backup_keys_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _groq_keys(_settings(monkeypatch)) == ["groq-key"]


def test_the_main_key_comes_first_then_the_backups_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS="k2,k3,k4")
    assert _groq_keys(settings) == ["groq-key", "k2", "k3", "k4"]


def test_blanks_spaces_and_repeats_are_dropped_from_the_backup_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS=" k2, k3 ,,k2,groq-key,k4,")
    assert _groq_keys(settings) == ["groq-key", "k2", "k3", "k4"]


# --- the cached settings ---


def test_get_settings_loads_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # a folder without a .env file, so the real one is never read
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
