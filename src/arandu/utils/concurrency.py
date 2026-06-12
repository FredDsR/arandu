"""Bounded, rate-limit-adaptive concurrency for batch LLM stages.

The rag batch runners process independent records whose dominant cost is
a remote LLM call, so a small thread pool multiplies throughput against
a server with matching parallel slots (``OLLAMA_NUM_PARALLEL``). The
helper keeps all shared-state mutation in the caller: workers only run
the supplied function; results and errors are yielded back to the
consuming (main) thread, where checkpoint writes and file saves stay
single-threaded and lock-free.

Hosted providers (Gemini, OpenAI) enforce per-minute quotas. When a call
exhausts the client-level retry budget on rate limits, the batch adapts
instead of dropping concurrency: :class:`AdaptiveThrottle` halves the
in-flight limit (AIMD, never below 1), pauses new submissions for a
cooldown, requeues the rate-limited item, and additively recovers the
limit after sustained successes.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_RATE_LIMIT_REQUEUES = 3
# Submission window per worker: bounds in-memory futures without starving the pool.
_WINDOW_FACTOR = 4


class AdaptiveThrottle:
    """AIMD concurrency limiter shared by the workers of one batch run.

    Multiplicative decrease: a rate-limited release halves the limit
    (floor 1) and pauses all acquisition for ``cooldown_seconds``.
    Additive increase: every ``recovery_successes`` consecutive
    successful releases restore one slot, up to the starting maximum.
    Plain failures neither shrink nor recover.

    Attributes:
        current_limit: The number of calls currently allowed in flight.
    """

    def __init__(
        self,
        max_limit: int,
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        recovery_successes: int = 10,
    ) -> None:
        """Initialize at full concurrency.

        Args:
            max_limit: Starting and maximum in-flight limit; must be >= 1.
            cooldown_seconds: Pause applied to all acquisition after a
                rate-limited release.
            recovery_successes: Consecutive successes required to win
                back one slot.

        Raises:
            ValueError: If ``max_limit`` is not a positive integer.
        """
        if max_limit < 1:
            raise ValueError(f"max_limit must be >= 1, got {max_limit}")
        self._max = max_limit
        self._limit = max_limit
        self._in_flight = 0
        self._successes = 0
        self._pause_until = 0.0
        self._cooldown = cooldown_seconds
        self._recovery = recovery_successes
        self._cv = threading.Condition()

    @property
    def current_limit(self) -> int:
        """Return the number of calls currently allowed in flight."""
        with self._cv:
            return self._limit

    def acquire(self) -> None:
        """Block until a slot is free and any cooldown has elapsed."""
        with self._cv:
            while True:
                remaining = self._pause_until - time.monotonic()
                if remaining > 0:
                    self._cv.wait(timeout=remaining)
                    continue
                if self._in_flight >= self._limit:
                    self._cv.wait()
                    continue
                self._in_flight += 1
                return

    def release_success(self) -> None:
        """Release a slot after a successful call; may recover the limit."""
        with self._cv:
            self._in_flight -= 1
            self._successes += 1
            if self._successes >= self._recovery and self._limit < self._max:
                self._limit += 1
                self._successes = 0
            self._cv.notify_all()

    def release_rate_limited(self) -> None:
        """Release a slot after a rate limit: halve the limit and cool down."""
        with self._cv:
            self._in_flight -= 1
            self._limit = max(1, self._limit // 2)
            self._successes = 0
            self._pause_until = time.monotonic() + self._cooldown
            self._cv.notify_all()

    def release_failure(self) -> None:
        """Release a slot after a non-rate-limit failure (no adaptation)."""
        with self._cv:
            self._in_flight -= 1
            self._cv.notify_all()


def map_concurrent[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    rate_limit_of: Callable[[Exception], bool] | None = None,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    max_rate_limit_requeues: int = DEFAULT_RATE_LIMIT_REQUEUES,
) -> Iterator[tuple[T, R | None, Exception | None]]:
    """Map ``fn`` over ``items`` with bounded, adaptive thread concurrency.

    Yields ``(item, result, error)`` triples; exactly one of ``result`` /
    ``error`` is non-``None`` per triple. With ``workers == 1`` the items
    run inline on the calling thread in submission order (no pool, no
    overhead). With ``workers > 1`` triples arrive in completion order.

    When ``rate_limit_of`` is provided, an error it classifies as a rate
    limit triggers adaptation instead of failing the item: the shared
    :class:`AdaptiveThrottle` halves the in-flight limit and cools down,
    and the item is requeued up to ``max_rate_limit_requeues`` times
    before its error is finally yielded. Non-rate-limit errors are never
    requeued (per-record isolation is the caller's contract).

    Args:
        fn: The per-item work. Exceptions it raises are captured and
            yielded, never propagated.
        items: The work items. Consumed lazily through a bounded
            submission window (``workers * 4`` futures at most).
        workers: Maximum simultaneous calls; must be >= 1.
        rate_limit_of: Optional predicate classifying an exception as a
            provider rate limit (e.g. exhausted 429 retries).
        cooldown_seconds: Pause applied after each rate-limit event.
        max_rate_limit_requeues: Re-attempts granted to a rate-limited
            item before its error is yielded.

    Yields:
        One ``(item, result, error)`` triple per input item.

    Raises:
        ValueError: If ``workers`` is not a positive integer.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    if workers == 1:
        yield from _map_inline(
            fn,
            items,
            rate_limit_of=rate_limit_of,
            cooldown_seconds=cooldown_seconds,
            max_rate_limit_requeues=max_rate_limit_requeues,
        )
        return

    throttle = (
        AdaptiveThrottle(workers, cooldown_seconds=cooldown_seconds)
        if rate_limit_of is not None
        else None
    )

    def run(item: T) -> R:
        if throttle is None:
            return fn(item)
        throttle.acquire()
        try:
            result = fn(item)
        except Exception as exc:
            if rate_limit_of is not None and rate_limit_of(exc):
                throttle.release_rate_limited()
            else:
                throttle.release_failure()
            raise
        throttle.release_success()
        return result

    # Bounded submission window: never more than workers * _WINDOW_FACTOR
    # futures alive at once, so memory stays flat however large the item
    # list grows (a full-corpus batch is tens of thousands of records).
    window = workers * _WINDOW_FACTOR
    items_iter = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[object, tuple[T, int]] = {}

        def refill() -> None:
            while len(pending) < window:
                try:
                    item = next(items_iter)
                except StopIteration:
                    return
                pending[pool.submit(run, item)] = (item, 0)

        refill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item, requeues = pending.pop(future)
                try:
                    yield item, future.result(), None
                except Exception as exc:
                    if _is_requeueable(exc, rate_limit_of, requeues, max_rate_limit_requeues):
                        pending[pool.submit(run, item)] = (item, requeues + 1)
                    else:
                        yield item, None, exc
            refill()


def _is_requeueable(
    exc: Exception,
    rate_limit_of: Callable[[Exception], bool] | None,
    requeues_done: int,
    max_requeues: int,
) -> bool:
    """Single source of truth for the requeue decision (both execution paths)."""
    return rate_limit_of is not None and rate_limit_of(exc) and requeues_done < max_requeues


def _map_inline[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    rate_limit_of: Callable[[Exception], bool] | None,
    cooldown_seconds: float,
    max_rate_limit_requeues: int,
) -> Iterator[tuple[T, R | None, Exception | None]]:
    """Sequential path: submission order, with the same requeue semantics."""
    for item in items:
        requeues_done = 0
        while True:
            try:
                yield item, fn(item), None
                break
            except Exception as exc:
                if _is_requeueable(exc, rate_limit_of, requeues_done, max_rate_limit_requeues):
                    requeues_done += 1
                    if cooldown_seconds > 0:
                        time.sleep(cooldown_seconds)
                    continue
                yield item, None, exc
                break
