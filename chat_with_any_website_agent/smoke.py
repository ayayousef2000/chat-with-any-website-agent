"""A first check that Weaviate, Cohere and Groq work with the keys and limits in the settings.

It makes a handful of real calls, so it is started by hand, once, after the keys are in ``.env``. Weaviate is only
read, unless ``--prepare`` asks to create the collection and the tenants. Cohere gets one embedding and one rerank
request. Every Groq key gets one tiny request, so that each account is proven on its own, and the judge model gets
one. Every call is counted, and a key never appears in the output. ``--dry-run`` shows what would be done and calls
nothing.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import weaviate
from groq import Groq
from pydantic import ValidationError
from weaviate.classes.init import AdditionalConfig, Auth, Timeout

from chat_with_any_website_agent.cohere_support import COHERE_RETRYABLE
from chat_with_any_website_agent.config import get_settings
from chat_with_any_website_agent.embeddings import CohereEmbedder
from chat_with_any_website_agent.groq_client import GROQ_RETRYABLE, GroqClient
from chat_with_any_website_agent.reranking import CohereReranker
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy
from chat_with_any_website_agent.usage import UsageCounter
from chat_with_any_website_agent.vector_store import WeaviateStore, open_tenant

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chat_with_any_website_agent.config import Settings

ENVIRONMENTS = ("development", "eval", "demo")
SERVICES = ("weaviate", "cohere", "groq")

GROQ_PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
# The models used here reason before they answer, so the reply needs room, and little reasoning keeps it cheap.
GROQ_OPTIONS = {"max_completion_tokens": 512, "reasoning_effort": "low", "temperature": 0}

QUESTION = "What is a cat?"
RELEVANT_TEXT = "A cat is a small domesticated animal that is often kept as a pet."
OTHER_TEXT = "The stock market closed higher on Tuesday after a quiet session."


@dataclass(frozen=True)
class Check:
    """The result of one check.

    Attributes:
        service: The service that was checked.
        name: What was checked.
        ok: Whether it worked.
        detail: What was found, or why it failed.
    """

    service: str
    name: str
    ok: bool
    detail: str


def _describe(error: BaseException) -> str:
    """Say what went wrong without the details of a library, which are long and can hold a request."""
    message = str(error).strip().splitlines()[0][:200] if str(error).strip() else ""
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def check_weaviate(client: Any, collection_name: str, tenant: str, *, prepare: bool) -> list[Check]:
    """Check the Weaviate cluster, the collection and the tenants.

    Args:
        client: A connected Weaviate client.
        collection_name: The name of the collection.
        tenant: The tenant of the environment in use, whose chunks are counted.
        prepare: Create the collection and the tenants of the three environments when they are missing.

    Returns:
        The checks, in order. Nothing is written unless ``prepare`` is true.
    """
    try:
        ready = client.is_ready()
    except Exception as error:
        return [Check("weaviate", "connection", False, _describe(error))]
    if not ready:
        return [Check("weaviate", "connection", False, "the cluster is not ready; it may be waking up, try again soon")]
    checks = [Check("weaviate", "connection", True, "the cluster is ready")]
    try:
        if prepare:
            for environment in ENVIRONMENTS:
                open_tenant(client, collection_name, environment)
        if not client.collections.exists(collection_name):
            detail = f"{collection_name} does not exist yet; run with --prepare to create it"
            return [*checks, Check("weaviate", "collection", True, detail)]
        collection = client.collections.use(collection_name)
        if not collection.config.get().multi_tenancy_config.enabled:
            return [*checks, Check("weaviate", "collection", False, f"{collection_name} is not multi-tenant")]
        tenants = sorted(collection.tenants.get())
        detail = f"{collection_name} is multi-tenant; tenants: {', '.join(tenants) or 'none yet'}"
        if tenant in tenants:
            count = WeaviateStore(collection.with_tenant(tenant)).count_chunks()
            detail += f"; {count} chunk(s) in {tenant}"
        checks.append(Check("weaviate", "collection", True, detail))
    except Exception as error:
        checks.append(Check("weaviate", "collection", False, _describe(error)))
    return checks


def check_cohere(embedder: Any, reranker: Any, dimension: int) -> list[Check]:
    """Check the embedding and the reranking of Cohere with one request each.

    Args:
        embedder: A ``CohereEmbedder``.
        reranker: A ``CohereReranker``.
        dimension: The size of the vectors the settings ask for.

    Returns:
        Two checks.
    """
    checks: list[Check] = []
    try:
        vector = embedder.embed_query(QUESTION)
        checks.append(Check("cohere", "embedding", len(vector) == dimension, f"one vector of size {len(vector)}"))
    except Exception as error:
        checks.append(Check("cohere", "embedding", False, _describe(error)))
    try:
        ranked = reranker.rerank(QUESTION, [RELEVANT_TEXT, OTHER_TEXT], top_n=2)
        best = ranked[0]
        checks.append(Check("cohere", "reranking", best.index == 0, f"the relevant text came first: {best.index == 0}"))
    except Exception as error:
        checks.append(Check("cohere", "reranking", False, _describe(error)))
    return checks


def check_groq(keys: Sequence[Any], judge: Any, judge_model: str) -> list[Check]:
    """Check every Groq key with one tiny request, and the judge model with one more.

    Args:
        keys: One client for each key, each using that key only.
        judge: A client for the judge model.
        judge_model: The name of the judge model, for the report.

    Returns:
        One check for each key, then one for the judge model.
    """
    checks: list[Check] = []
    for number, client in enumerate(keys, start=1):
        checks.append(_ask_groq(client, f"key {number}"))
    checks.append(_ask_groq(judge, f"judge model {judge_model}"))
    return checks


def _ask_groq(client: Any, name: str) -> Check:
    try:
        reply = client.complete(GROQ_PROMPT, **GROQ_OPTIONS)
    except Exception as error:
        return Check("groq", name, False, _describe(error))
    answered = bool(reply.text)
    detail = f"replied {reply.text[:30]!r}, {reply.total_tokens} tokens" if answered else "the reply was empty"
    return Check("groq", name, answered, detail)


def _one_attempt(retryable: tuple[type[BaseException], ...]) -> Retrier:
    """A check should show the first failure, not hide it behind retries."""
    return Retrier(retryable, RetryPolicy(attempts=1))


def run_checks(
    settings: Settings, usage: UsageCounter, *, prepare: bool = False, services: Sequence[str] = SERVICES
) -> list[Check]:
    """Build the clients from the settings and run the checks of the chosen services.

    Args:
        settings: The settings, with the keys.
        usage: Counts every call.
        prepare: Create the collection and the tenants in Weaviate when they are missing.
        services: The services to check.

    Returns:
        All the checks. A service that cannot even be set up gives a failed check instead of an error.
    """
    checks: list[Check] = []
    if "weaviate" in services:
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.weaviate_url,
                auth_credentials=Auth.api_key(settings.weaviate_api_key.get_secret_value()),
                additional_config=AdditionalConfig(timeout=Timeout(init=settings.weaviate_init_timeout_seconds)),
            )
        except Exception as error:
            checks.append(Check("weaviate", "connection", False, _describe(error)))
        else:
            try:
                checks.extend(
                    check_weaviate(client, settings.weaviate_collection, settings.weaviate_tenant, prepare=prepare)
                )
            finally:
                client.close()
    if "cohere" in services:
        retrier = _one_attempt(COHERE_RETRYABLE)
        embedder = CohereEmbedder(
            settings.cohere_api_key,
            settings.cohere_embed_model,
            settings.cohere_embed_dimension,
            retrier=retrier,
            usage=usage,
        )
        reranker = CohereReranker(settings.cohere_api_key, settings.cohere_rerank_model, retrier=retrier, usage=usage)
        checks.extend(check_cohere(embedder, reranker, settings.cohere_embed_dimension))
    if "groq" in services:
        retrier = _one_attempt(GROQ_RETRYABLE)
        keys = [
            GroqClient(
                [Groq(api_key=key.get_secret_value(), max_retries=0)],
                settings.groq_model,
                settings.groq_tokens_per_minute,
                retrier=retrier,
                usage=usage,
            )
            for key in settings.groq_keys
        ]
        judge = GroqClient.from_settings(settings, model=settings.eval_judge_model, usage=usage, retrier=retrier)
        checks.extend(check_groq(keys, judge, settings.eval_judge_model))
    return checks


def secret_values(settings: Settings) -> list[str]:
    """Every key in the settings, so that none of them can reach the output."""
    return [
        settings.cohere_api_key.get_secret_value(),
        settings.weaviate_api_key.get_secret_value(),
        *(key.get_secret_value() for key in settings.groq_keys),
    ]


def format_report(checks: Sequence[Check], secrets: Sequence[str] = ()) -> str:
    """Write the checks as lines of text, with every secret value replaced.

    Args:
        checks: The results.
        secrets: The values that must never be shown.

    Returns:
        One line for each check, and a last line that says how many passed.
    """
    lines = [
        f"[{'ok' if check.ok else 'FAILED'}]".ljust(10) + f"{check.service:<10}{check.name:<26}{check.detail}"
        for check in checks
    ]
    failed = sum(not check.ok for check in checks)
    lines.append("")
    lines.append(f"All {len(checks)} checks passed." if not failed else f"{failed} of {len(checks)} checks failed.")
    text = "\n".join(lines)
    for secret in filter(None, secrets):
        text = text.replace(secret, "***")
    return text


def describe_plan(settings: Settings, services: Sequence[str], *, prepare: bool) -> str:
    """Say what a run would do, without doing it.

    Args:
        settings: The settings.
        services: The services that would be checked.
        prepare: Whether the collection and the tenants would be created.

    Returns:
        A short description with the number of calls.
    """
    lines = ["This check would make these calls, and write nothing unless --prepare is given:"]
    if "weaviate" in services:
        action = (
            f"create {settings.weaviate_collection} and the tenants {', '.join(ENVIRONMENTS)} if missing"
            if prepare
            else f"read {settings.weaviate_collection} and its tenants (nothing is written)"
        )
        lines.append(f"  Weaviate: connect, then {action}.")
    if "cohere" in services:
        lines.append(
            f"  Cohere: 2 calls, one embedding ({settings.cohere_embed_model}) and one rerank "
            f"({settings.cohere_rerank_model})."
        )
    if "groq" in services:
        keys = len(settings.groq_keys)
        lines.append(
            f"  Groq: {keys + 1} calls, one for each of {keys} key(s) with {settings.groq_model} and one with "
            f"{settings.eval_judge_model} (a few hundred tokens each at most)."
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-smoke", description=__doc__)
    parser.add_argument("--only", nargs="+", choices=SERVICES, default=list(SERVICES), help="Check only these")
    parser.add_argument("--prepare", action="store_true", help="Create the Weaviate collection and the tenants")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done and call nothing")
    return parser


def _say(text: str, *, error: bool = False) -> None:
    (sys.stderr if error else sys.stdout).write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke check from the command line.

    Args:
        argv: The arguments, without the program name. They are read from the command line when ``None``.

    Returns:
        0 when every check passed, 1 when one failed, 2 when the settings are not valid.
    """
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as error:
        # Only the names and the reasons are shown, never the values.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']).upper() or 'SETTINGS'}: {item['msg']}"
            for item in error.errors()
        )
        _say(f"The settings are not valid ({problems}). Fill in .env first.", error=True)
        return 2
    _say(describe_plan(settings, args.only, prepare=args.prepare))
    if args.dry_run:
        return 0
    usage = UsageCounter()
    checks = run_checks(settings, usage, prepare=args.prepare, services=args.only)
    _say("")
    _say(format_report(checks, secret_values(settings)))
    _say(f"Calls counted: {usage.calls}. Tokens counted: {usage.tokens}.")
    return 0 if all(check.ok for check in checks) else 1
