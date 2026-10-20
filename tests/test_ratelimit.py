"""Tests of the limits per visitor, with a clock that the tests move."""

import pytest
from starlette.requests import Request

from chat_with_any_website_agent import ratelimit
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.errors import TooManyRequestsError
from chat_with_any_website_agent.ratelimit import ClientLimits, RateLimiter, Rule, client_id, describe_wait

MINUTE = Rule(3, 60, "too fast, wait {wait}")
DAY = Rule(5, 86_400, "too many today, wait {wait}")


class Clock:
    """A time that only moves when a test says so."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _request(host: str | None = "203.0.113.7", forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    scope = {"type": "http", "headers": headers, "client": (host, 5000) if host is not None else None}
    return Request(scope)


# --- the sliding window ---


def test_uses_up_to_the_limit_are_allowed_and_the_next_is_refused_with_its_wait() -> None:
    clock = Clock()
    limiter = RateLimiter({"ask": [MINUTE]}, clock)

    assert [limiter.check("ask", "a") for _ in range(3)] == [None, None, None]
    clock.now += 20
    refusal = limiter.check("ask", "a")

    assert refusal is not None
    assert refusal.rule is MINUTE
    assert refusal.wait == pytest.approx(40)  # the first use leaves the window 60 s after it was made


def test_a_use_is_allowed_again_once_the_oldest_use_has_left_the_window() -> None:
    clock = Clock()
    limiter = RateLimiter({"ask": [MINUTE]}, clock)
    for _ in range(3):
        limiter.check("ask", "a")

    clock.now += 59
    assert limiter.check("ask", "a") is not None
    clock.now += 2
    assert limiter.check("ask", "a") is None


def test_a_refused_use_is_not_counted() -> None:
    clock = Clock()
    limiter = RateLimiter({"ask": [MINUTE]}, clock)
    for _ in range(3):
        limiter.check("ask", "a")
    for _ in range(10):  # trying again and again does not push the end of the wait further away
        assert limiter.check("ask", "a") is not None

    clock.now += 61
    assert limiter.check("ask", "a") is None


def test_clients_and_actions_are_counted_separately() -> None:
    limiter = RateLimiter({"ask": [MINUTE], "load": [MINUTE]}, Clock())
    for _ in range(3):
        limiter.check("ask", "a")

    assert limiter.check("ask", "a") is not None
    assert limiter.check("ask", "b") is None
    assert limiter.check("load", "a") is None


def test_an_action_without_rules_is_never_refused() -> None:
    limiter = RateLimiter({"ask": [MINUTE]}, Clock())

    assert all(limiter.check("other", "a") is None for _ in range(50))


def test_with_two_rules_the_one_with_the_longest_wait_is_reported() -> None:
    clock = Clock()
    limiter = RateLimiter({"ask": [MINUTE, DAY]}, clock)
    for _ in range(5):  # five uses, spread out so that the minute rule is not reached
        assert limiter.check("ask", "a") is None
        clock.now += 30
    clock.now += 60  # now both windows contain uses; the day rule is at its limit

    refusal = limiter.check("ask", "a")

    assert refusal is not None
    assert refusal.rule is DAY
    assert refusal.wait > 80_000


def test_the_minute_rule_is_reported_when_only_it_is_reached() -> None:
    limiter = RateLimiter({"ask": [MINUTE, DAY]}, Clock())
    for _ in range(3):
        limiter.check("ask", "a")

    refusal = limiter.check("ask", "a")

    assert refusal is not None
    assert refusal.rule is MINUTE


def test_only_the_most_recently_seen_clients_are_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ratelimit, "MAX_CLIENTS", 3)
    clock = Clock()
    limiter = RateLimiter({"ask": [MINUTE]}, clock)

    for number in range(6):
        clock.now += 1
        limiter.check("ask", f"client-{number}")

    assert len(limiter._uses) == 3
    assert ("ask", "client-5") in limiter._uses
    assert ("ask", "client-0") not in limiter._uses


def test_a_client_that_is_seen_again_is_not_forgotten_before_one_that_was_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ratelimit, "MAX_CLIENTS", 2)
    clock = Clock()
    limiter = RateLimiter({"ask": [DAY]}, clock)
    for name in ("first", "second"):
        clock.now += 1
        limiter.check("ask", name)
    clock.now += 1
    limiter.check("ask", "first")  # "first" is the most recently seen now
    clock.now += 1
    limiter.check("ask", "third")

    assert set(limiter._uses) == {("ask", "first"), ("ask", "third")}


# --- wording ---


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "about 1 second"),
        (0.4, "about 1 second"),
        (20.2, "about 20 seconds"),
        (90, "about 90 seconds"),
        (91, "about 2 minutes"),
        (600, "about 10 minutes"),
        (5400, "about 90 minutes"),
        (5401, "about 2 hours"),
        (40_000, "about 11 hours"),
    ],
)
def test_waits_are_described_in_words(seconds: float, text: str) -> None:
    assert describe_wait(seconds) == text


# --- who is calling ---


def test_the_client_is_the_address_of_the_caller() -> None:
    assert client_id(_request("203.0.113.7")) == "203.0.113.7"


def test_the_forwarded_header_is_never_read_because_anyone_can_send_one() -> None:
    assert client_id(_request("10.0.0.1", forwarded="198.51.100.9")) == "10.0.0.1"
    assert client_id(_request("10.0.0.1", forwarded="6.6.6.6, 198.51.100.9")) == "10.0.0.1"


def test_ipv6_addresses_of_one_visitor_count_as_one_client() -> None:
    first = client_id(_request("2001:db8:1:2:aaaa:bbbb:cccc:dddd"))
    second = client_id(_request("2001:db8:1:2:1111:2222:3333:4444"))
    other = client_id(_request("2001:db8:1:3::1"))

    assert first == second == "2001:db8:1:2::/64"
    assert other != first


def test_a_host_that_is_not_an_address_is_kept_as_it_is() -> None:
    assert client_id(_request("testclient")) == "testclient"
    assert client_id(_request(None)) == "unknown"


# --- the limits of the server ---


def test_client_limits_refuse_with_a_message_and_a_wait() -> None:
    limits = ClientLimits(asks_per_minute=1, asks_per_day=10, loads_per_hour=1, clock=Clock())
    request = _request()
    limits.enforce(request, "ask")

    with pytest.raises(TooManyRequestsError) as refused:
        limits.enforce(request, "ask")

    assert str(refused.value) == "You're asking questions too quickly. Please try again in about 60 seconds."
    assert refused.value.retry_after == pytest.approx(60)


def test_the_daily_limit_has_its_own_message() -> None:
    limits = ClientLimits(asks_per_minute=100, asks_per_day=2, loads_per_hour=1, clock=Clock())
    request = _request()
    limits.enforce(request, "ask")
    limits.enforce(request, "ask")

    with pytest.raises(TooManyRequestsError, match=r"today's limit of questions.*24 hours"):
        limits.enforce(request, "ask")


def test_page_loads_have_their_own_message_and_do_not_use_up_questions() -> None:
    limits = ClientLimits(asks_per_minute=1, asks_per_day=10, loads_per_hour=1, clock=Clock())
    request = _request()
    limits.enforce(request, "load")

    with pytest.raises(TooManyRequestsError, match="loaded a lot of pages"):
        limits.enforce(request, "load")
    limits.enforce(request, "ask")  # questions are counted apart from loads


def test_limits_that_are_switched_off_never_refuse() -> None:
    limits = ClientLimits(asks_per_minute=1, asks_per_day=1, loads_per_hour=1, enabled=False)

    for _ in range(20):
        limits.enforce(_request(), "ask")
        limits.enforce(_request(), "load")


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    env = {
        "COHERE_API_KEY": "cohere-key",
        "WEAVIATE_URL": "https://example.weaviate.cloud",
        "WEAVIATE_API_KEY": "weaviate-key",
        "GROQ_API_KEY": "groq-key",
        **overrides,
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_limits_are_built_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, RATE_LIMIT_ASKS_PER_MINUTE="1", RATE_LIMIT_ASKS_PER_DAY="5")
    limits = ClientLimits.from_settings(settings)
    request = _request()

    limits.enforce(request, "ask")
    with pytest.raises(TooManyRequestsError):
        limits.enforce(request, "ask")


def test_limits_can_be_switched_off_in_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, RATE_LIMIT_ENABLED="false", RATE_LIMIT_ASKS_PER_MINUTE="1")
    limits = ClientLimits.from_settings(settings)

    for _ in range(5):
        limits.enforce(_request(), "ask")
