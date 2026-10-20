"""The deployment files describe a server that exists: real settings, a real start function, a working proxy list."""

import asyncio
import ipaddress
import re
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from chat_with_any_website_agent import web
from chat_with_any_website_agent.config import Settings

ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
SERVICES: dict[str, dict[str, Any]] = {service["name"]: service for service in BLUEPRINT["services"]}
PRODUCTION = SERVICES["chat-with-any-website-agent"]
STAGING = SERVICES["chat-with-any-website-agent-staging"]


def _variables(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in service["envVars"]}


# Variables that Render's services set for programs other than the settings of the agent.
NOT_SETTINGS = {"FORWARDED_ALLOW_IPS"}  # read by uvicorn

EACH_SERVICE = pytest.mark.parametrize("service", [PRODUCTION, STAGING], ids=["production", "staging"])


def test_the_blueprint_describes_a_production_and_a_staging_service() -> None:
    assert sorted(SERVICES) == ["chat-with-any-website-agent", "chat-with-any-website-agent-staging"]


@EACH_SERVICE
def test_each_service_is_a_free_web_service_built_from_the_dockerfile(service: dict[str, Any]) -> None:
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["dockerfilePath"] == "./Dockerfile"
    assert (ROOT / "Dockerfile").is_file()
    assert service["plan"] == "free"
    assert service["autoDeployTrigger"] == "commit"


def test_production_deploys_the_released_branch_and_staging_the_one_that_is_tried_first() -> None:
    assert PRODUCTION["branch"] == "main"
    assert STAGING["branch"] == "develop"


@EACH_SERVICE
def test_the_health_check_is_the_path_that_answers_without_calling_a_service(service: dict[str, Any]) -> None:
    assert service["healthCheckPath"] == web.HEALTH_PATH


@EACH_SERVICE
def test_every_variable_of_a_service_is_a_setting_or_read_by_uvicorn(service: dict[str, Any]) -> None:
    settings = {name.upper() for name in Settings.model_fields}

    unknown = {name for name in _variables(service) if name not in settings | NOT_SETTINGS}

    assert unknown == set()


@EACH_SERVICE
def test_the_keys_and_addresses_are_asked_for_and_never_written_in_the_file(service: dict[str, Any]) -> None:
    variables = _variables(service)

    for name in ("COHERE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY", "GROQ_API_KEY", "GROQ_BACKUP_API_KEYS"):
        assert variables[name] == {"key": name, "sync": False}, name
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert not re.search(r"(?i)(api[_-]?key|secret|token)\s*:\s*['\"]?[A-Za-z0-9]{16,}", text)


@pytest.mark.parametrize(("service", "tenant"), [(PRODUCTION, "production"), (STAGING, "staging")])
def test_the_values_of_a_service_are_accepted_by_the_settings(
    monkeypatch: pytest.MonkeyPatch, service: dict[str, Any], tenant: str
) -> None:
    for name, value in {
        "COHERE_API_KEY": "k",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "k",
        "GROQ_API_KEY": "k",
    }.items():
        monkeypatch.setenv(name, value)
    for name, item in _variables(service).items():
        if "value" in item:
            monkeypatch.setenv(name, str(item["value"]))

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.weaviate_tenant == tenant
    assert settings.weaviate_collection == "WebsiteChunk"
    assert settings.rate_limit_enabled is True
    assert settings.web_max_stored_chunks == 20_000
    assert settings.web_host_names == frozenset()  # the name of the service is added from RENDER_EXTERNAL_HOSTNAME


def test_the_two_services_keep_their_pages_apart_and_apart_from_the_tenants_of_a_computer() -> None:
    tenants = [_variables(service)["WEAVIATE_TENANT"]["value"] for service in (PRODUCTION, STAGING)]

    assert len(set(tenants)) == 2
    assert not set(tenants) & {"development", "eval", "demo"}


def test_both_services_have_the_same_limits_and_trusted_proxies() -> None:
    same = (
        "RATE_LIMIT_ENABLED",
        "RATE_LIMIT_ASKS_PER_MINUTE",
        "RATE_LIMIT_ASKS_PER_DAY",
        "RATE_LIMIT_LOADS_PER_HOUR",
        "WEB_MAX_STORED_CHUNKS",
        "FORWARDED_ALLOW_IPS",
        "WEAVIATE_COLLECTION",
    )

    for name in same:
        assert _variables(PRODUCTION)[name] == _variables(STAGING)[name], name


# --- the visitor behind Render's proxies ---


def _client_seen_through(trusted: str, peer: str, forwarded: str) -> str:
    """The address the server takes for the caller, after uvicorn read the forwarded header."""
    seen: list[str] = []

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        seen.append(scope["client"][0])

    middleware = ProxyHeadersMiddleware(app, trusted_hosts=trusted)  # type: ignore[arg-type]
    scope = {"type": "http", "client": (peer, 5000), "headers": [(b"x-forwarded-for", forwarded.encode())]}
    asyncio.run(middleware(scope, None, None))  # type: ignore[arg-type]
    return seen[0]


TRUSTED = _variables(PRODUCTION)["FORWARDED_ALLOW_IPS"]["value"]


def test_the_trusted_addresses_are_valid_networks() -> None:
    for item in TRUSTED.split(","):
        ipaddress.ip_network(item.strip(), strict=True)


def test_the_visitor_is_the_first_address_from_the_right_that_is_not_a_proxy() -> None:
    seen = _client_seen_through(TRUSTED, peer="10.20.30.40", forwarded="198.51.100.9, 172.70.34.5, 10.20.30.41")

    assert seen == "198.51.100.9"


def test_an_address_that_a_caller_forges_in_the_header_cannot_become_the_visitor() -> None:
    seen = _client_seen_through(
        TRUSTED, peer="10.20.30.40", forwarded="6.6.6.6, 198.51.100.9, 172.70.34.5, 10.20.30.41"
    )

    assert seen == "198.51.100.9"  # what the caller wrote sits to the left of the real visitor


def test_a_header_from_an_address_that_is_not_a_proxy_is_not_believed() -> None:
    seen = _client_seen_through(TRUSTED, peer="203.0.113.50", forwarded="198.51.100.9")

    assert seen == "203.0.113.50"


# --- the image ---


def test_the_image_starts_the_function_that_builds_the_application_from_the_environment() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    command = re.search(r"^CMD (.+)$", dockerfile, re.MULTILINE)

    assert command is not None
    assert "--factory chat_with_any_website_agent.web:create_app_from_environment" in command.group(1)
    assert "${PORT:-8000}" in command.group(1)
    assert "--host 0.0.0.0" in command.group(1)
    assert callable(web.create_app_from_environment)


def test_the_health_check_of_the_image_asks_the_health_path() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert web.HEALTH_PATH in dockerfile


def test_the_image_runs_as_an_unprivileged_user_and_copies_what_it_needs() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert re.search(r"^USER app$", dockerfile, re.MULTILINE)
    for needed in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "chat_with_any_website_agent"):
        assert needed in dockerfile
        assert (ROOT / needed).exists(), needed
    assert ".env" not in re.findall(r"^COPY .*$", dockerfile, re.MULTILINE)


def test_the_build_context_leaves_out_everything_except_what_the_dockerfile_copies() -> None:
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    allowed = {line.removeprefix("!") for line in lines if line.startswith("!")}

    assert lines[0] == "*"  # an allowlist: a new local file is left out without being named
    assert allowed == {"pyproject.toml", "uv.lock", "README.md", "LICENSE", "chat_with_any_website_agent/"}
    assert not any(".env" in name for name in allowed)
