"""Shared test fixtures."""

import pytest

from chat_with_any_website_agent.config import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without any of the settings in the environment, whatever the developer's shell holds."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
