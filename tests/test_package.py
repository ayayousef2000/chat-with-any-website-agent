"""Smoke test: the package imports and reports a valid version."""

import re

import chat_with_any_website_agent


def test_package_has_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", chat_with_any_website_agent.__version__)
