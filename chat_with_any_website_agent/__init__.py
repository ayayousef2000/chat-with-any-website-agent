"""Chat With Any Website Agent: a LangGraph workflow for grounded, cited answers about web pages and PDFs."""

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

__version__ = "0.1.0"

# Two libraries that this project uses call a function of Python that will be removed in Python 3.16, and print a
# notice about it every time they are imported or a client is made. The notice is about their code and there is
# nothing to change here, so only that notice from those two libraries is hidden.
#
# It is hidden where warnings are shown, not with a warning filter: another library (Weaviate) puts a filter that
# shows every warning in front of any filter set here, whenever it is imported.
_NOTICE = re.compile(r"'asyncio\.iscoroutinefunction' is deprecated")
_LIBRARIES = frozenset({"cohere", "langsmith"})


class NoticeFilter:
    """Shows warnings like ``warnings.showwarning`` does, except the notice of the two libraries about Python."""

    # The mark by which the filter is recognized. A class would not do: loading the package again makes a new one.
    hides_known_notice: ClassVar[bool] = True

    def __init__(self, show: Callable[..., None]) -> None:
        """Create the filter.

        Args:
            show: What shows a warning, which is called for every warning except the hidden notice.
        """
        self._show = show

    @property
    def wrapped(self) -> Callable[..., None]:
        """What shows the warnings that are not hidden."""
        return self._show

    def __call__(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        """Show a warning, unless it is the notice of one of the two libraries about the function of Python."""
        known = (
            issubclass(category, DeprecationWarning)
            and _NOTICE.match(str(message)) is not None
            and not _LIBRARIES.isdisjoint(Path(filename).parts)  # it was raised by a file of one of the libraries
        )
        if not known:
            self._show(message, category, filename, lineno, file, line)


def _hide_known_notice() -> None:
    """Put the filter in place of ``warnings.showwarning``, once."""
    if not getattr(warnings.showwarning, "hides_known_notice", False):
        vars(warnings)["showwarning"] = NoticeFilter(warnings.showwarning)


_hide_known_notice()
