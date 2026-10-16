"""The environment template and the settings describe the same variables."""

from pathlib import Path
from typing import TYPE_CHECKING

from chat_with_any_website_agent.config import Settings

if TYPE_CHECKING:
    import pytest

TEMPLATE = Path(__file__).resolve().parent.parent / ".env.example"


def _template_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            values[name.strip()] = value.strip()
    return values


def test_every_variable_in_the_template_is_a_setting() -> None:
    assert {name.lower() for name in _template_values()} <= set(Settings.model_fields)


def test_every_setting_is_in_the_template() -> None:
    assert set(Settings.model_fields) <= {name.lower() for name in _template_values()}


def test_the_values_in_the_template_are_the_defaults() -> None:
    for name, value in _template_values().items():
        if value:
            assert str(Settings.model_fields[name.lower()].default) == value, name


def test_the_template_is_accepted_once_the_blank_required_values_are_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    required = {name for name, field in Settings.model_fields.items() if field.is_required()}
    for name, value in _template_values().items():
        if value:
            monkeypatch.setenv(name, value)
        elif name.lower() in required:
            monkeypatch.setenv(name, "placeholder")
    assert Settings(_env_file=None).weaviate_tenant == "development"  # type: ignore[call-arg]
