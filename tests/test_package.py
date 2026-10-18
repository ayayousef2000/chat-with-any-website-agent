"""Smoke test: the package imports and reports a valid version."""

import importlib
import re
import warnings

import chat_with_any_website_agent

NOTICE = "'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect instead"


def test_package_has_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", chat_with_any_website_agent.__version__)


def test_only_one_notice_of_two_libraries_is_hidden() -> None:
    # The filter is added when the package is imported, and the test run discards what imports add, so import again.
    importlib.reload(chat_with_any_website_agent)
    hiding = [
        module
        for action, message, category, module, _ in warnings.filters
        if action == "ignore"
        and category is DeprecationWarning
        and message is not None
        and module is not None
        and message.match(NOTICE)
    ]

    assert hiding, "the notice is not hidden"
    assert all(module.match("cohere.client") or module.match("langsmith._internal") for module in hiding)
    assert not any(module.match("chat_with_any_website_agent.graph") for module in hiding)  # our own code is not hidden
