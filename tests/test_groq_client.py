"""Tests of the Groq client with fake Groq clients and a fake clock: nothing is sent and nothing waits."""

from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from chat_with_any_website_agent import groq_client
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import BudgetExceededError, ServiceBusyError
from chat_with_any_website_agent.groq_client import (
    GROQ_RETRYABLE,
    ChatReply,
    GroqClient,
    estimate_tokens,
    patient_retrier,
)
from chat_with_any_website_agent.retrying import Retrier, RetryPolicy
from chat_with_any_website_agent.usage import GROQ, UsageCounter

_REQUEST = httpx.Request("POST", "https://api.example.invalid/v1")
MESSAGES = [{"role": "user", "content": "hi"}]


def _limit(retry_after: str | None) -> groq.RateLimitError:
    headers = {} if retry_after is None else {"retry-after": retry_after}
    response = httpx.Response(429, request=_REQUEST, headers=headers)
    return groq.RateLimitError("limit", response=response, body=None)


class Keys:
    """Fake Groq clients that record their use; the ones named in ``limited`` answer "too many requests"."""

    def __init__(self, count: int = 3, wait: str | None = "30", prompt: int = 10, completion: int = 5) -> None:
        self.used: list[int] = []
        self.calls: list[dict[str, Any]] = []
        self.limited: set[int] = set()
        self.waits: dict[int, str | None] = dict.fromkeys(range(count), wait)
        self.failures: list[BaseException] = []
        self.tokens = (prompt, completion)
        self.clients = [self._client(number) for number in range(count)]

    def _client(self, number: int) -> Any:
        def create(**kwargs: Any) -> Any:
            self.used.append(number)
            self.calls.append(kwargs)
            if self.failures:
                raise self.failures.pop(0)
            if number in self.limited:
                raise _limit(self.waits[number])
            message = SimpleNamespace(content=f"  Answer {number}  ")
            usage = SimpleNamespace(prompt_tokens=self.tokens[0], completion_tokens=self.tokens[1])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class Clock:
    """A clock that only moves when something sleeps. ``waits`` are the pauses of the pace, ``pauses`` of retries."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds


def _client(
    keys: Keys,
    clock: Clock | None = None,
    *,
    policy: RetryPolicy | None = None,
    usage: UsageCounter | None = None,
    tokens_per_minute: int = 1_000_000,
) -> tuple[GroqClient, Clock]:
    clock = clock or Clock()
    retrier = Retrier(GROQ_RETRYABLE, policy or RetryPolicy(), sleep=clock.pause, jitter=lambda _: 0.0)
    client = GroqClient(
        keys.clients, "model-x", tokens_per_minute, retrier=retrier, usage=usage, clock=clock, sleep=clock.wait
    )
    return client, clock


def _ask(client: GroqClient) -> str:
    return client.complete(MESSAGES).text


# --- the reply ---


def test_the_reply_carries_the_text_the_tokens_and_the_number_of_the_key() -> None:
    keys = Keys(prompt=120, completion=30)
    client, _ = _client(keys)

    reply = client.complete(MESSAGES, temperature=0, response_format={"type": "json_object"})

    assert reply == ChatReply(text="Answer 0", prompt_tokens=120, completion_tokens=30, key_number=1)
    assert reply.total_tokens == 150
    call = keys.calls[0]
    assert call["model"] == "model-x"
    assert call["messages"] == MESSAGES
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}


def test_a_reply_without_usage_information_counts_no_tokens() -> None:
    keys = Keys()
    client, _ = _client(keys)
    create = keys.clients[0].chat.completions.create
    keys.clients[0].chat.completions.create = lambda **kwargs: SimpleNamespace(
        choices=create(**kwargs).choices, usage=None
    )

    assert client.complete(MESSAGES).total_tokens == 0


def test_a_patient_retrier_waits_for_a_limit_to_lift_longer_than_the_default() -> None:
    retrier = patient_retrier()

    assert retrier.retryable == GROQ_RETRYABLE
    assert retrier.policy == RetryPolicy(attempts=5, max_delay=60.0, budget=240.0)
    assert retrier.policy.budget > RetryPolicy().budget


def test_the_client_names_the_model_it_asks() -> None:
    client, _ = _client(Keys())

    assert client.model == "model-x"


def test_at_least_one_client_is_needed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GroqClient([], "model-x", 1000)


def test_the_size_of_a_request_is_estimated_from_its_text_and_its_reply_allowance() -> None:
    assert estimate_tokens([{"content": "a" * 30}, {"content": None}, {}], 100) == 110


# --- several keys, used in turn ---


def test_the_first_key_is_used_until_its_limit_is_reached() -> None:
    keys = Keys()
    client, _ = _client(keys)

    assert [_ask(client) for _ in range(3)] == ["Answer 0"] * 3
    assert keys.used == [0, 0, 0]


def test_each_key_takes_over_when_the_one_before_is_limited() -> None:
    keys = Keys(count=6)
    client, _ = _client(keys)

    keys.limited = {0}
    assert _ask(client) == "Answer 1"
    assert keys.used == [0, 1]  # the request that hit the limit moved on at once, with no waiting

    keys.used.clear()
    assert _ask(client) == "Answer 1"
    assert keys.used == [1]  # the first key is resting, so it is not asked again

    keys.used.clear()
    keys.limited = {0, 1}
    assert _ask(client) == "Answer 2"
    assert keys.used == [1, 2]

    keys.used.clear()
    keys.limited = {0, 1, 2, 3, 4}
    assert _ask(client) == "Answer 5"
    assert keys.used == [2, 3, 4, 5]  # the third key is limited now, as are the fourth and fifth; the sixth answers

    keys.used.clear()
    assert _ask(client) == "Answer 5"
    assert keys.used == [5]  # every earlier key is resting, so only the sixth is asked


def test_after_the_last_key_the_first_is_used_again_once_it_has_recovered() -> None:
    keys = Keys(count=3)
    client, clock = _client(keys)

    keys.limited = {0, 1}
    assert _ask(client) == "Answer 2"
    assert keys.used == [0, 1, 2]

    keys.used.clear()
    keys.limited = {2}
    clock.now = 31.0  # the first and second keys have recovered
    assert _ask(client) == "Answer 0"
    assert keys.used == [2, 0]  # the circle went on from the third key to the first


def test_a_key_that_has_recovered_is_not_returned_to_while_the_current_one_still_works() -> None:
    keys = Keys(count=3)
    client, clock = _client(keys)

    keys.limited = {0}
    assert _ask(client) == "Answer 1"
    keys.limited = set()
    clock.now = 31.0  # the first key has recovered, but the second one is in use and not limited
    keys.used.clear()
    assert [_ask(client) for _ in range(2)] == ["Answer 1"] * 2
    assert keys.used == [1, 1]


def test_a_key_with_no_stated_wait_rests_for_a_minute() -> None:
    keys = Keys(count=2, wait=None)
    client, clock = _client(keys, policy=RetryPolicy(attempts=1))

    keys.limited = {0}
    assert _ask(client) == "Answer 1"
    keys.used.clear()
    keys.limited = {1}
    clock.now = 59.0
    with pytest.raises(ServiceBusyError):
        _ask(client)
    assert keys.used == [1]  # the first key had rested for less than a minute, so it was not asked

    keys.used.clear()
    keys.limited = set()
    clock.now = 61.0
    assert _ask(client) == "Answer 0"  # a minute has passed for the first key, the second one still rests
    assert keys.used == [0]


def test_when_every_key_is_limited_the_shortest_wait_is_reported() -> None:
    keys = Keys(count=3)
    keys.waits = {0: "40", 1: "25", 2: "90"}
    keys.limited = {0, 1, 2}
    client, clock = _client(keys)

    with pytest.raises(ServiceBusyError) as raised:
        _ask(client)

    assert keys.used == [0, 1, 2]  # all three were tried before giving up
    assert raised.value.retry_after == 25.0
    assert clock.pauses == []  # 25 seconds is longer than a person should wait


def test_a_key_that_is_still_resting_is_skipped_in_the_circle() -> None:
    keys = Keys(count=3)
    client, clock = _client(keys)

    keys.limited = {0, 1}
    assert _ask(client) == "Answer 2"
    keys.used.clear()
    keys.limited = {2}
    clock.now = 10.0  # the first and second keys are still resting
    with pytest.raises(ServiceBusyError):
        _ask(client)
    assert keys.used == [2]


def test_a_request_that_waits_continues_with_a_key_that_has_recovered() -> None:
    keys = Keys(count=2, wait="10")
    client, clock = _client(keys)
    original = keys.clients[0].chat.completions.create
    calls = [0]

    def first_key(**kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            keys.limited = {0, 1}  # both keys are limited on the first try, then they recover
        elif calls[0] == 2:
            keys.limited = set()
        return original(**kwargs)

    keys.clients[0].chat.completions.create = first_key

    assert _ask(client) == "Answer 0"
    assert clock.pauses == [10.0]


def test_closing_the_client_closes_every_groq_client_that_can_be_closed() -> None:
    keys = Keys(count=3)
    closed: list[int] = []
    for number in (0, 2):  # the second client has no close method at all
        keys.clients[number].close = lambda number=number: closed.append(number)
    client, _ = _client(keys)

    client.close()

    assert closed == [0, 2]


# --- the pace of each key ---


def _long_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "x" * 300}]  # about 100 tokens, 110 with a reply allowance of 10


def test_a_request_waits_when_the_only_key_has_no_room_in_the_minute() -> None:
    keys = Keys(count=1)
    client, clock = _client(keys, tokens_per_minute=100)

    client.complete(_long_messages(), max_completion_tokens=10)
    client.complete(_long_messages(), max_completion_tokens=10)

    assert clock.waits == [60.0]
    assert keys.used == [0, 0]


def test_a_key_with_room_is_used_instead_of_waiting_for_the_key_in_use() -> None:
    keys = Keys(count=2)
    client, clock = _client(keys, tokens_per_minute=200)

    client.complete(_long_messages(), max_completion_tokens=10)
    client.complete(_long_messages(), max_completion_tokens=10)

    assert keys.used == [0, 1]
    assert clock.waits == []


def test_the_real_use_corrects_the_estimate() -> None:
    keys = Keys(count=1, prompt=600, completion=395)  # 995 tokens, far more than the estimate of 10
    client, clock = _client(keys, tokens_per_minute=1000)

    client.complete(MESSAGES, max_completion_tokens=10)
    client.complete(MESSAGES, max_completion_tokens=10)

    assert clock.waits == [60.0]


def test_a_key_that_waited_for_its_place_and_is_then_limited_ends_in_a_busy_error() -> None:
    keys = Keys(count=1)
    client, clock = _client(keys, tokens_per_minute=100)
    client.complete(_long_messages(), max_completion_tokens=10)  # the key has no room left in this minute
    keys.limited = {0}

    with pytest.raises(ServiceBusyError) as raised:
        client.complete(_long_messages(), max_completion_tokens=10)

    assert clock.waits == [60.0]  # it waited for its place, and then Groq said its limit was reached
    assert raised.value.retry_after == 30.0


# --- counting ---


def test_every_attempt_is_counted_including_the_one_that_was_limited() -> None:
    usage = UsageCounter()
    keys = Keys()
    client, _ = _client(keys, usage=usage)
    keys.limited = {0}

    _ask(client)

    assert usage.calls == {GROQ: 2}
    assert usage.tokens == {GROQ: 15}


def test_a_run_stops_at_its_limit_of_calls() -> None:
    usage = UsageCounter(max_calls={GROQ: 2})
    keys = Keys()
    client, _ = _client(keys, usage=usage)
    _ask(client)
    _ask(client)

    with pytest.raises(BudgetExceededError):
        _ask(client)

    assert keys.used == [0, 0]


def test_a_run_stops_at_its_limit_of_tokens_but_the_reply_that_crossed_it_is_delivered() -> None:
    usage = UsageCounter(max_tokens={GROQ: 20})
    keys = Keys()  # 15 tokens a reply
    client, _ = _client(keys, usage=usage)

    assert _ask(client) == "Answer 0"
    assert _ask(client) == "Answer 0"  # 30 tokens now, over the limit of 20, but this reply is still delivered
    with pytest.raises(BudgetExceededError):
        _ask(client)

    assert keys.used == [0, 0]
    assert usage.tokens == {GROQ: 30}


# --- limits, outages and errors ---


@pytest.mark.parametrize(
    "error",
    [
        groq.APIConnectionError(request=_REQUEST),
        groq.APITimeoutError(request=_REQUEST),
        groq.InternalServerError("oops", response=httpx.Response(503, request=_REQUEST), body=None),
    ],
    ids=["connection", "timeout", "server"],
)
def test_a_connection_that_fails_or_a_server_error_is_tried_again(error: BaseException) -> None:
    keys = Keys()
    keys.failures = [error]
    client, clock = _client(keys)

    assert _ask(client) == "Answer 0"
    assert clock.pauses == [1.0]


def test_when_groq_stays_unreachable_the_caller_gets_a_busy_error() -> None:
    keys = Keys()
    keys.failures = [groq.APIConnectionError(request=_REQUEST)] * 4
    client, _ = _client(keys)

    with pytest.raises(ServiceBusyError) as raised:
        _ask(client)

    assert str(raised.value) == "Groq is busy right now. Try again in a few seconds."


def test_an_error_that_retrying_cannot_fix_reaches_the_caller_at_once() -> None:
    keys = Keys()
    keys.failures = [groq.AuthenticationError("bad key", response=httpx.Response(401, request=_REQUEST), body=None)]
    client, clock = _client(keys)

    with pytest.raises(groq.AuthenticationError):
        _ask(client)

    assert (keys.used, clock.pauses) == ([0], [])


# --- building the client from the settings ---


def test_the_client_is_built_from_the_settings_with_one_groq_client_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class FakeGroq:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.sent: dict[str, Any] = {}
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            created.append(self)

        def create(self, **kwargs: Any) -> Any:
            self.sent = kwargs
            message = SimpleNamespace(content="hello")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    monkeypatch.setattr(groq_client, "Groq", FakeGroq)
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "w")
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_BACKUP_API_KEYS", "k2, k3 ,,k1")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    client = GroqClient.from_settings(settings)

    expected = [{"api_key": key, "max_retries": 0} for key in ("k1", "k2", "k3")]
    assert [instance.kwargs for instance in created] == expected
    assert client.complete(MESSAGES).text == "hello"
    assert created[0].sent["model"] == "openai/gpt-oss-120b"

    judge = GroqClient.from_settings(settings, model="judge-model")  # another model, for example a judge

    assert judge.complete(MESSAGES).text == "hello"
    assert created[3].sent["model"] == "judge-model"  # the first key of the second client
