"""Tests of the throttle that keeps requests below a limit per window."""

import pytest

from chat_with_any_website_agent.throttle import Throttle


class Clock:
    """A clock that only moves when something sleeps, so the tests never really wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _throttle(limit: int, window: float = 60.0) -> tuple[Throttle, Clock]:
    clock = Clock()
    return Throttle(limit, window, clock=clock, sleep=clock.sleep), clock


def test_requests_within_the_limit_do_not_wait() -> None:
    throttle, clock = _throttle(3)

    assert [throttle.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert clock.slept == []


def test_a_request_over_the_limit_waits_until_the_oldest_leaves_the_window() -> None:
    throttle, clock = _throttle(2, window=60)
    throttle.acquire()
    clock.now = 20.0
    throttle.acquire()
    clock.now = 25.0

    waited = throttle.acquire()

    assert waited == 35.0  # the first request is 60 seconds old at t = 60
    assert clock.now == 60.0


def test_the_window_slides_so_old_requests_stop_counting() -> None:
    throttle, clock = _throttle(2, window=10)
    throttle.acquire()
    throttle.acquire()
    clock.now = 10.0

    assert throttle.acquire() == 0.0
    assert throttle.acquire() == 0.0
    assert clock.slept == []


def test_units_count_towards_the_limit() -> None:
    throttle, clock = _throttle(100, window=60)
    throttle.acquire(60)
    clock.now = 10.0

    assert throttle.acquire(30) == 0.0  # 90 of 100 used
    assert throttle.acquire(30) == 50.0  # waits for the 60 units used at t = 0 to leave the window


def test_a_request_larger_than_the_whole_limit_goes_through_when_nothing_else_is_counted() -> None:
    throttle, clock = _throttle(10)

    assert throttle.acquire(50) == 0.0
    assert clock.slept == []


def test_a_request_larger_than_the_whole_limit_waits_for_the_window_to_empty() -> None:
    throttle, _ = _throttle(10, window=60)
    throttle.acquire(1)

    assert throttle.acquire(50) == 60.0


def test_a_request_that_fits_is_counted_without_waiting() -> None:
    throttle, clock = _throttle(100)

    assert throttle.try_acquire(60) is True
    assert throttle.try_acquire(40) is True
    assert throttle.try_acquire(1) is False  # the limit is reached, and nothing was waited for
    assert clock.slept == []


def test_a_request_that_does_not_fit_is_not_counted() -> None:
    throttle, clock = _throttle(100, window=60)
    throttle.try_acquire(100)
    clock.now = 30.0

    assert throttle.try_acquire(10) is False
    clock.now = 60.0

    assert throttle.try_acquire(100) is True  # only the first 100 was counted, and it has left the window


def test_units_that_were_used_are_counted_even_beyond_the_limit() -> None:
    throttle, clock = _throttle(100, window=60)
    throttle.acquire(10)
    throttle.record(150)  # the real use of the request turned out to be much more than the estimate

    assert throttle.acquire(10) == 60.0
    assert clock.now == 60.0


@pytest.mark.parametrize(("limit", "window"), [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)])
def test_a_limit_or_window_that_makes_no_sense_is_refused(limit: int, window: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        Throttle(limit, window)
