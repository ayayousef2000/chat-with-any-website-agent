"""The command line of the evaluation: ``agent-eval validate``, ``summary`` and ``run``.

``validate`` and ``summary`` read files only and call no service. ``run`` answers and judges the questions of a set
with a system, and stops cleanly at the call and token limits in the settings; it is the only command that uses Cohere
and Groq, so it is started by hand.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from chat_with_any_website_agent.config import get_settings
from chat_with_any_website_agent.errors import DatasetError
from chat_with_any_website_agent.evaluation import evaluation_usage, load_cases, load_results, run_and_save
from chat_with_any_website_agent.evaluation_summary import format_summary, summarize
from chat_with_any_website_agent.groq_client import GROQ_RETRYABLE, GroqClient
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy
from chat_with_any_website_agent.systems import SYSTEMS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chat_with_any_website_agent.evaluation import Case

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_UNKNOWN_SYSTEM = 2
EXIT_STOPPED = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: The arguments, without the program name. They are read from the command line when ``None``.

    Returns:
        The exit code: 0 for success, 1 for a file that cannot be used, 2 for an unknown system and 3 when a run
        stopped early (it can be continued with ``--resume``).
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.datasets)
        if args.command == "summary":
            return _summary(args.results)
        return _run(args)
    except DatasetError as error:
        _say(str(error), error=True)
        return EXIT_BAD_INPUT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Check question sets and count their questions (no service used)")
    validate.add_argument("datasets", nargs="+", type=Path)

    summary = commands.add_parser("summary", help="Summarize results files (no service is used)")
    summary.add_argument("results", nargs="+", type=Path)

    run = commands.add_parser("run", help="Answer and judge a question set with a system (uses Cohere and Groq)")
    run.add_argument("dataset", type=Path)
    run.add_argument("--system", required=True, help="The system to evaluate")
    run.add_argument("--out", required=True, type=Path, help="The results file, saved after every question")
    run.add_argument("--resume", action="store_true", help="Continue the results already in the file")
    run.add_argument("--limit", type=int, help="Answer at most this many more questions, for a small first run")
    return parser


def _say(text: str, *, error: bool = False) -> None:
    (sys.stderr if error else sys.stdout).write(text + "\n")


def _count(cases: Sequence[Case], attribute: str) -> str:
    counts = Counter(getattr(case, attribute) for case in cases)
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))


def _validate(datasets: Sequence[Path]) -> int:
    for dataset in datasets:
        cases = load_cases(dataset)
        unanswerable = sum(not case.answerable for case in cases)
        _say(f"{dataset}: {len(cases)} questions ({len(cases) - unanswerable} answerable, {unanswerable} not)")
        _say(f"  categories: {_count(cases, 'category')}")
        _say(f"  languages: {_count(cases, 'language')}")
    return EXIT_OK


def _summary(files: Sequence[Path]) -> int:
    results = [result for file in files for result in load_results(file)]
    _say(format_summary(summarize(results)))
    return EXIT_OK


def _run(args: argparse.Namespace) -> int:
    factory = SYSTEMS.get(args.system)
    if factory is None:
        _say(f"Unknown system {args.system!r}. Available: {', '.join(sorted(SYSTEMS)) or 'none yet'}.", error=True)
        return EXIT_UNKNOWN_SYSTEM
    cases = load_cases(args.dataset)
    settings = get_settings()
    usage = evaluation_usage(settings)
    # The judge is patient: a run is allowed to wait for a limit to lift.
    judge = GroqClient.from_settings(
        settings,
        model=settings.eval_judge_model,
        usage=usage,
        retrier=Retrier(GROQ_RETRYABLE, RetryPolicy(attempts=5, max_delay=60.0, budget=240.0)),
    )
    system = factory(settings, usage)
    try:
        report = run_and_save(system, cases, judge, usage, out=args.out, resume=args.resume, limit=args.limit)
    finally:
        system.close()
    _say(format_summary(summarize(report.results)))
    _say(f"\nThis run: {report.evaluated} question(s). Calls: {usage.calls}. Tokens: {usage.tokens}.")
    _say(f"Results are in {args.out}.")
    if report.stopped_because:
        _say(f"\nThe run stopped early: {report.stopped_because} Continue later with --resume.", error=True)
        return EXIT_STOPPED
    return EXIT_OK
