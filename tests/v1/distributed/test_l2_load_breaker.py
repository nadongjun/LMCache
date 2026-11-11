# SPDX-License-Identifier: Apache-2.0
"""Tests for the L2 load breaker and its command line plumbing."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.distributed.storage_controllers.l2_load_breaker import (
    MIN_SAMPLES_TO_OPEN,
    L2LoadBreaker,
    L2LoadBreakerState,
)

# 1 GB/s floor; loads below are slow.
FLOOR_GBPS = 1.0
FAST = (2_000_000_000, 1.0)  # 2 GB/s
SLOW = (100_000_000, 1.0)  # 0.1 GB/s
OPEN_SECONDS = 30.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_breaker(clock: FakeClock, floor_gbps: float = FLOOR_GBPS) -> L2LoadBreaker:
    return L2LoadBreaker("mock[0]", floor_gbps, OPEN_SECONDS, clock=clock)


def record(breaker: L2LoadBreaker, request_id: int, sample: tuple[int, float]) -> None:
    breaker.record_load(request_id, sample[0], sample[1])


def trip(breaker: L2LoadBreaker) -> None:
    """Record enough slow loads to open a closed breaker."""
    for request_id in range(MIN_SAMPLES_TO_OPEN):
        record(breaker, request_id, SLOW)
    assert breaker.state == L2LoadBreakerState.OPEN


def test_slow_loads_open_only_after_min_samples(clock):
    breaker = make_breaker(clock)
    for request_id in range(MIN_SAMPLES_TO_OPEN - 1):
        record(breaker, request_id, SLOW)
        assert breaker.state == L2LoadBreakerState.CLOSED

    record(breaker, MIN_SAMPLES_TO_OPEN, SLOW)

    assert breaker.state == L2LoadBreakerState.OPEN
    assert not breaker.admit(1000)


def test_occasional_slow_load_does_not_open(clock):
    breaker = make_breaker(clock)
    for request_id in range(4 * MIN_SAMPLES_TO_OPEN):
        record(breaker, request_id, SLOW if request_id % 4 == 0 else FAST)

    assert breaker.state == L2LoadBreakerState.CLOSED


def test_zero_floor_never_opens(clock):
    breaker = make_breaker(clock, floor_gbps=0.0)
    for request_id in range(4 * MIN_SAMPLES_TO_OPEN):
        record(breaker, request_id, (1, 10.0))

    assert breaker.state == L2LoadBreakerState.CLOSED
    assert breaker.admit(1000)


def test_admits_one_probe_after_open_seconds(clock):
    breaker = make_breaker(clock)
    trip(breaker)

    clock.now += OPEN_SECONDS - 1
    assert not breaker.admit(1)

    clock.now += 1
    assert breaker.admit(2)
    assert breaker.state == L2LoadBreakerState.HALF_OPEN
    assert breaker.allows(2)
    assert not breaker.admit(3)
    assert not breaker.allows(3)


def test_fast_probe_closes(clock):
    breaker = make_breaker(clock)
    trip(breaker)
    clock.now += OPEN_SECONDS
    assert breaker.admit(1)

    record(breaker, 1, FAST)

    assert breaker.state == L2LoadBreakerState.CLOSED
    assert breaker.admit(2)


def test_slow_probe_reopens_for_another_period(clock):
    breaker = make_breaker(clock)
    trip(breaker)
    clock.now += OPEN_SECONDS
    assert breaker.admit(1)

    record(breaker, 1, SLOW)

    assert breaker.state == L2LoadBreakerState.OPEN
    clock.now += OPEN_SECONDS - 1
    assert not breaker.admit(2)
    clock.now += 1
    assert breaker.admit(3)


def test_probe_without_a_load_passes_the_probe_on(clock):
    breaker = make_breaker(clock)
    trip(breaker)
    clock.now += OPEN_SECONDS
    assert breaker.admit(1)
    assert not breaker.admit(2)

    breaker.release(1)

    assert breaker.admit(3)
    assert breaker.state == L2LoadBreakerState.HALF_OPEN


def test_loads_of_requests_admitted_before_opening_are_ignored(clock):
    breaker = make_breaker(clock)
    trip(breaker)
    clock.now += OPEN_SECONDS
    assert breaker.admit(1)

    # A request admitted while closed finishes its fast load late.
    record(breaker, 0, FAST)
    assert breaker.state == L2LoadBreakerState.HALF_OPEN

    record(breaker, 1, SLOW)
    assert breaker.state == L2LoadBreakerState.OPEN


def test_reopening_after_recovery_needs_min_samples_again(clock):
    breaker = make_breaker(clock)
    trip(breaker)
    clock.now += OPEN_SECONDS
    assert breaker.admit(1)
    record(breaker, 1, FAST)

    for request_id in range(2, MIN_SAMPLES_TO_OPEN):
        record(breaker, request_id, SLOW)
        assert breaker.state == L2LoadBreakerState.CLOSED
    record(breaker, MIN_SAMPLES_TO_OPEN, SLOW)
    assert breaker.state == L2LoadBreakerState.OPEN


# =============================================================================
# Command line plumbing
# =============================================================================

_BASE_ARGS = ["--l1-size-gb", "1", "--eviction-policy", "noop"]


def test_breaker_is_off_by_default():
    config = parse_args(_BASE_ARGS)
    assert config.l2_load_min_throughput_gbps == 0.0
    assert config.l2_load_breaker_open_seconds == 30.0


def test_breaker_flags_are_plumbed():
    config = parse_args(
        _BASE_ARGS
        + [
            "--l2-load-min-throughput-gbps",
            "1.5",
            "--l2-load-breaker-open-seconds",
            "10",
        ]
    )
    assert config.l2_load_min_throughput_gbps == 1.5
    assert config.l2_load_breaker_open_seconds == 10.0


@pytest.mark.parametrize(
    ("flag", "value", "field"),
    [
        ("--l2-load-min-throughput-gbps", "-1", "l2_load_min_throughput_gbps"),
        ("--l2-load-breaker-open-seconds", "0", "l2_load_breaker_open_seconds"),
    ],
)
def test_out_of_range_flags_are_rejected(flag, value, field):
    with pytest.raises(ValueError, match=field):
        parse_args(_BASE_ARGS + [flag, value])
