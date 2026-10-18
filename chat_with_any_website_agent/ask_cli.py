"""The command ``agent-ask``: ask a question about a web page or PDF file and see the cited answer and the steps taken.

It loads the source (a stored copy is reused), runs the agent and prints the answer, the excerpts it cites and a line
for each step the agent took. It uses Cohere, Weaviate and Groq, so it is started by hand.
"""

import argparse
import sys
import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from chat_with_any_website_agent.config import describe_validation_error, get_settings
from chat_with_any_website_agent.errors import (
    BudgetExceededError,
    ExtractionError,
    FetchError,
    InvalidURLError,
    ServiceBusyError,
)
from chat_with_any_website_agent.smoke import secret_values
from chat_with_any_website_agent.usage import COHERE, GROQ, UsageCounter
from chat_with_any_website_agent.wiring import build_runtime

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import timedelta

    from chat_with_any_website_agent.graph import AgentAnswer
    from chat_with_any_website_agent.ingest import IngestResult

EXIT_OK = 0
EXIT_BAD_SOURCE = 1
EXIT_BAD_SETTINGS = 2
EXIT_SERVICE = 3


def _ago(age: timedelta) -> str:
    """Say how long ago something was, in the largest whole unit: days, hours or minutes."""
    seconds = int(age.total_seconds())
    for unit, size in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        if seconds >= size:
            count = seconds // size
            return f"{count} {unit}{'' if count == 1 else 's'} ago"
    return "less than a minute ago"


def _describe(result: IngestResult, name: str) -> str:
    """Say what happened to a source: loaded, reused as it was, checked and unchanged, or loaded again."""
    stored = f"{result.chunk_count} chunk(s) stored"
    if result.changed:
        return f"The page of {name} changed since the stored copy was made, so it was loaded again ({stored})."
    if not result.reused:
        again = (
            " The stored copy was too old, or its age or version was not known, so it was loaded again."
            if result.expired
            else ""
        )
        return f"Loaded {name} ({stored}).{again}"
    loaded = f" (loaded {_ago(result.age)})" if result.age is not None else ""
    if result.problem:
        asked = f"Could not check whether the page of {name} changed ({result.problem})"
        return f"{asked}, so the stored copy{loaded} was used."
    if result.checked:
        return f"Checked {name}: the page has not changed since the stored copy was made{loaded}."
    return f"Reused the stored copy of {name}{loaded}."


def format_answer(
    ingested: IngestResult,
    answer: AgentAnswer,
    usage: UsageCounter,
    seconds: float,
    *,
    show_steps: bool = True,
    extras: Sequence[IngestResult] = (),
) -> str:
    """Write the answer, its sources, the steps the agent took and what the question cost.

    Args:
        ingested: What happened to the source.
        answer: The answer of the agent.
        usage: The counter that counted the calls and tokens.
        seconds: How long the question took, including the loading of the sources.
        show_steps: Whether to list the steps of the agent.
        extras: What happened to the further sources that were searched as well.

    Returns:
        The text to print.
    """
    lines = [_describe(ingested, "the source")]
    lines += [_describe(extra, extra.url) for extra in extras]
    lines += ["", answer.answer]
    if answer.abstained:
        lines += ["", f"Reason: {answer.reason}"]
        if answer.draft:
            lines += ["", "The answer that the check rejected:", f"  {answer.draft}"]
    if answer.citations:
        lines += ["", "Sources cited:"]
        for citation in answer.citations:
            where = f"{citation.title} > {citation.heading}" if citation.heading else citation.title
            lines.append(f"  [{citation.number}] {where or citation.url} ({citation.url})")
    if show_steps:
        lines += ["", "Steps:"]
        lines += [f"  {number}. {step}" for number, step in enumerate(answer.steps, start=1)]
    calls, tokens = usage.calls, usage.tokens
    lines += [
        "",
        f"Cohere calls: {calls.get(COHERE, 0)}. Groq calls: {calls.get(GROQ, 0)}. "
        f"Groq tokens: {tokens.get(GROQ, 0)}. Time: {seconds:.1f} s.",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-ask", description=__doc__)
    parser.add_argument("url", help="The address of a web page or PDF file")
    parser.add_argument("question", help="The question, in quotes")
    parser.add_argument(
        "--also", action="append", default=[], metavar="URL", help="A further source to search as well; can be repeated"
    )
    parser.add_argument("--tenant", help="The Weaviate tenant to use instead of WEAVIATE_TENANT, for example demo")
    parser.add_argument(
        "--refresh", action="store_true", help="Download the source again even if a copy of it is stored"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Use a stored copy without asking the website whether the page changed",
    )
    parser.add_argument("--no-steps", action="store_true", help="Do not list the steps of the agent")
    return parser


def _say(text: str, *, error: bool = False) -> None:
    (sys.stderr if error else sys.stdout).write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: The arguments, without the program name. They are read from the command line when ``None``.

    Returns:
        0 when the question was answered or the agent said that the source does not cover it, 1 when the source
        could not be loaded, 2 when the settings are not valid and 3 when a service failed or a limit was reached.
    """
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as error:
        _say(f"The settings are not valid ({describe_validation_error(error)}). Fill in .env first.", error=True)
        return EXIT_BAD_SETTINGS
    usage = UsageCounter()
    started = time.monotonic()
    try:
        runtime = build_runtime(settings, usage, tenant=args.tenant)
        try:
            check = not args.no_check
            ingested = runtime.ingestor.ingest(args.url, refresh=args.refresh, check=check)
            extras = [runtime.ingestor.ingest(url, refresh=args.refresh, check=check) for url in args.also]
            answer = runtime.agent.ask(args.question, [ingested.url, *(extra.url for extra in extras)])
        finally:
            runtime.close()
    except (InvalidURLError, FetchError, ExtractionError) as error:
        _say(str(error), error=True)
        return EXIT_BAD_SOURCE
    except (ServiceBusyError, BudgetExceededError) as error:
        _say(str(error), error=True)
        return EXIT_SERVICE
    except Exception as error:  # a service that cannot be reached: say what kind of failure, never a key
        text = f"A service failed: {type(error).__name__}: {error}"
        for secret in filter(None, secret_values(settings)):
            text = text.replace(secret, "***")
        _say(text, error=True)
        return EXIT_SERVICE
    seconds = time.monotonic() - started
    _say(format_answer(ingested, answer, usage, seconds, show_steps=not args.no_steps, extras=extras))
    return EXIT_OK
