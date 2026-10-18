"""Smoke test: the package imports and reports a valid version, and hides one notice of two libraries."""

import importlib
import re
import warnings
from typing import Any, cast

import chat_with_any_website_agent
from chat_with_any_website_agent import NoticeFilter

NOTICE = "'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect instead"
COHERE = "C:/project/.venv/Lib/site-packages/cohere/client.py"
LANGSMITH = "C:/project/.venv/Lib/site-packages/langsmith/_internal/_beta_decorator.py"
OUR_OWN = "C:/project/chat_with_any_website_agent/graph.py"


class Recorder:
    """Stands in for what shows warnings, and records what it was asked to show."""

    def __init__(self) -> None:
        self.shown: list[tuple[Any, ...]] = []

    def __call__(self, *details: Any) -> None:
        self.shown.append(details)


def test_package_has_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", chat_with_any_website_agent.__version__)


def test_the_notice_of_the_two_libraries_about_python_is_not_shown() -> None:
    recorder = Recorder()
    hook = NoticeFilter(recorder)

    hook(NOTICE, DeprecationWarning, COHERE, 43)
    hook(NOTICE, DeprecationWarning, LANGSMITH, 70)
    hook(Warning(NOTICE), DeprecationWarning, COHERE, 114)  # a warning can come as an object and not as a text

    assert recorder.shown == []


def test_every_other_warning_is_still_shown() -> None:
    recorder = Recorder()
    hook = NoticeFilter(recorder)

    hook(NOTICE, DeprecationWarning, OUR_OWN, 1)  # the same notice raised by our own code
    hook("another thing is deprecated", DeprecationWarning, COHERE, 2)  # another notice of the library
    hook(NOTICE, UserWarning, COHERE, 3)  # the same text, but not a deprecation
    hook("a warning", UserWarning, OUR_OWN, 4, None, "the line")

    assert [details[3] for details in recorder.shown] == [1, 2, 3, 4]
    assert recorder.shown[3] == ("a warning", UserWarning, OUR_OWN, 4, None, "the line")


def test_the_filter_is_put_in_place_once_however_often_the_package_is_loaded() -> None:
    # The test run puts back the original function around every test, so load the package to put the filter in place.
    importlib.reload(chat_with_any_website_agent)
    first = cast("Any", warnings.showwarning)
    importlib.reload(chat_with_any_website_agent)

    assert type(first).__name__ == "NoticeFilter"
    assert not getattr(first.wrapped, "hides_known_notice", False)  # it does not wrap itself
    assert warnings.showwarning is first  # and the second load left it alone
