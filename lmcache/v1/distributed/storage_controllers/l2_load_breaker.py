# SPDX-License-Identifier: Apache-2.0
"""Skip an L2 adapter for prefetch while its loads are too slow to pay off.

Loading a prefix from L2 helps only while it is faster than recomputing the
prefix. The connection circuit breakers in the S3, Bigtable and Valkey
adapters open on failures; an adapter whose loads succeed but crawl (a
congested link, a far-away region, an overloaded backend) is never caught by
them and keeps holding requests in the lookup and load phases.

:class:`L2LoadBreaker` watches the load throughput of one adapter and opens
when its moving average falls below an operator-set floor. While it is open
the prefetch controller leaves the adapter out of new requests, so their
misses are recomputed. After ``open_seconds`` one request is let through as a
probe: a fast enough probe load closes the breaker, a slow one opens it again.

All methods run on the prefetch loop thread; ``state`` may be read from other
threads.
"""

# Standard
from collections.abc import Callable
import enum
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Weight of the newest sample in the moving average.
EWMA_ALPHA = 0.2

# Samples needed after closing before the breaker may open, so a few slow
# loads right after start or recovery do not trip it.
MIN_SAMPLES_TO_OPEN = 8


class L2LoadBreakerState(enum.IntEnum):
    """State of an :class:`L2LoadBreaker`; the values are exported as a gauge."""

    CLOSED = 0
    """The adapter serves prefetches."""

    OPEN = 1
    """The adapter is skipped by new prefetches."""

    HALF_OPEN = 2
    """One probe request may use the adapter; the others skip it."""


class L2LoadBreaker:
    """Throughput-based breaker for the prefetch loads of one L2 adapter.

    Args:
        name: The adapter name used in log messages.
        min_throughput_gbps: The load throughput floor in GB/s, measured as
            in ``lmcache_mp.l2_load_throughput`` (bytes over the time from
            submitting the load task to its completion). 0 disables the
            breaker: it never opens.
        open_seconds: How long the breaker stays open before letting a probe
            request through.
        clock: Monotonic clock in seconds; injectable for tests.
    """

    def __init__(
        self,
        name: str,
        min_throughput_gbps: float,
        open_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._min_bytes_per_s = min_throughput_gbps * 1e9
        self._open_seconds = open_seconds
        self._clock = clock

        self._state = L2LoadBreakerState.CLOSED
        self._ewma_bytes_per_s = 0.0
        self._num_samples = 0
        self._opened_at = 0.0
        self._probe_request_id: int | None = None

    @property
    def state(self) -> L2LoadBreakerState:
        """The current state."""
        return self._state

    def admit(self, request_id: int) -> bool:
        """Return whether a new prefetch request may look up the adapter.

        Once ``open_seconds`` have passed since the breaker opened, the first
        request asking becomes the probe and is admitted.

        Args:
            request_id: The prefetch request being started.

        Returns:
            True if the request may use the adapter.
        """
        if (
            self._state == L2LoadBreakerState.OPEN
            and self._clock() - self._opened_at >= self._open_seconds
        ):
            self._state = L2LoadBreakerState.HALF_OPEN
        if (
            self._state == L2LoadBreakerState.HALF_OPEN
            and self._probe_request_id is None
        ):
            self._probe_request_id = request_id
        return self.allows(request_id)

    def allows(self, request_id: int) -> bool:
        """Return whether an admitted request may still load from the adapter.

        Unlike :meth:`admit`, this never changes the state: a request admitted
        while the breaker was closed is dropped from the adapter's load plan
        if the breaker opened during its lookup.

        Args:
            request_id: The prefetch request being planned.

        Returns:
            True if the breaker is closed or ``request_id`` is the probe.
        """
        if self._state == L2LoadBreakerState.CLOSED:
            return True
        return request_id == self._probe_request_id

    def record_load(self, request_id: int, num_bytes: int, elapsed_s: float) -> None:
        """Record a finished load task of the adapter.

        While the breaker is closed the sample updates the moving average and
        may open it. While it is half-open only the probe's sample counts and
        decides whether it closes or opens again. Loads of requests admitted
        before the breaker opened are ignored once it is open.

        Args:
            request_id: The prefetch request the load belonged to.
            num_bytes: The bytes the load task was submitted for.
            elapsed_s: Seconds from submitting the task to its completion.
        """
        if self._min_bytes_per_s == 0:
            return
        bytes_per_s = num_bytes / elapsed_s

        if self._state == L2LoadBreakerState.CLOSED:
            if self._num_samples == 0:
                self._ewma_bytes_per_s = bytes_per_s
            else:
                self._ewma_bytes_per_s += EWMA_ALPHA * (
                    bytes_per_s - self._ewma_bytes_per_s
                )
            self._num_samples += 1
            if (
                self._num_samples >= MIN_SAMPLES_TO_OPEN
                and self._ewma_bytes_per_s < self._min_bytes_per_s
            ):
                self._open(self._ewma_bytes_per_s)
            return

        if (
            self._state == L2LoadBreakerState.HALF_OPEN
            and request_id == self._probe_request_id
        ):
            if bytes_per_s >= self._min_bytes_per_s:
                self._close(bytes_per_s)
            else:
                self._open(bytes_per_s)

    def release(self, request_id: int) -> None:
        """Forget a finished request.

        A probe that finishes without a load sample (its keys were all L1
        hits or L2 misses) frees the probe slot for the next request.

        Args:
            request_id: The prefetch request that finished.
        """
        if request_id == self._probe_request_id:
            self._probe_request_id = None

    def _open(self, bytes_per_s: float) -> None:
        if self._state == L2LoadBreakerState.CLOSED:
            logger.warning(
                "L2 adapter %s: load throughput %.3f GB/s is below the "
                "%.3f GB/s floor; skipping it for prefetch for %.0fs",
                self._name,
                bytes_per_s / 1e9,
                self._min_bytes_per_s / 1e9,
                self._open_seconds,
            )
        self._state = L2LoadBreakerState.OPEN
        self._opened_at = self._clock()
        self._probe_request_id = None

    def _close(self, bytes_per_s: float) -> None:
        logger.info(
            "L2 adapter %s: probe load throughput %.3f GB/s; using it for "
            "prefetch again",
            self._name,
            bytes_per_s / 1e9,
        )
        self._state = L2LoadBreakerState.CLOSED
        self._ewma_bytes_per_s = bytes_per_s
        self._num_samples = 1
        self._probe_request_id = None
