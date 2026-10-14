"""Smoke test: the package imports and reports its version."""

import chat_with_any_website_agent


def test_package_has_version() -> None:
    assert chat_with_any_website_agent.__version__ == "0.1.0"
