"""Tests for the bounded-concurrency mapping helper."""

from __future__ import annotations

import threading

import pytest

from arandu.utils.concurrency import map_concurrent


class TestMapConcurrent:
    def test_sequential_mode_preserves_submission_order(self) -> None:
        items = [3, 1, 2]
        triples = list(map_concurrent(lambda x: x * 10, items, workers=1))

        assert [t[0] for t in triples] == [3, 1, 2]
        assert [t[1] for t in triples] == [30, 10, 20]
        assert all(t[2] is None for t in triples)

    def test_concurrent_mode_processes_every_item(self) -> None:
        items = list(range(20))
        triples = list(map_concurrent(lambda x: x + 100, items, workers=4))

        assert sorted(t[0] for t in triples) == items
        assert sorted(t[1] for t in triples) == [x + 100 for x in items]
        assert all(t[2] is None for t in triples)

    def test_errors_are_yielded_not_raised(self) -> None:
        def boom(x: int) -> int:
            if x % 2 == 0:
                raise ValueError(f"bad {x}")
            return x

        triples = list(map_concurrent(boom, [1, 2, 3, 4], workers=2))

        ok = {t[0]: t[1] for t in triples if t[2] is None}
        errs = {t[0]: t[2] for t in triples if t[2] is not None}
        assert ok == {1: 1, 3: 3}
        assert set(errs) == {2, 4}
        assert all(isinstance(e, ValueError) for e in errs.values())
        assert all(t[1] is None for t in triples if t[2] is not None)

    def test_workers_run_simultaneously(self) -> None:
        # A barrier only releases when 3 calls are in flight at once, so
        # the test deadlocks (and times out) unless real parallelism exists.
        barrier = threading.Barrier(3)

        def wait_for_peers(x: int) -> int:
            barrier.wait(timeout=5)
            return x

        triples = list(map_concurrent(wait_for_peers, [1, 2, 3], workers=3))

        assert all(t[2] is None for t in triples)

    def test_sequential_mode_does_not_spawn_threads(self) -> None:
        main = threading.current_thread()
        seen: list[threading.Thread] = []

        def record_thread(x: int) -> int:
            seen.append(threading.current_thread())
            return x

        list(map_concurrent(record_thread, [1, 2], workers=1))

        assert all(t is main for t in seen)

    def test_rejects_nonpositive_workers(self) -> None:
        with pytest.raises(ValueError, match="workers"):
            list(map_concurrent(lambda x: x, [1], workers=0))


class FakeRateLimit(Exception):
    """Stand-in for a provider rate-limit error."""


def _is_fake_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, FakeRateLimit)


class TestAdaptiveThrottle:
    def test_starts_at_max_limit(self) -> None:
        from arandu.utils.concurrency import AdaptiveThrottle

        throttle = AdaptiveThrottle(4)
        assert throttle.current_limit == 4

    def test_rate_limit_halves_limit_never_below_one(self) -> None:
        from arandu.utils.concurrency import AdaptiveThrottle

        throttle = AdaptiveThrottle(4, cooldown_seconds=0.0)
        throttle.acquire()
        throttle.release_rate_limited()
        assert throttle.current_limit == 2
        throttle.acquire()
        throttle.release_rate_limited()
        assert throttle.current_limit == 1
        throttle.acquire()
        throttle.release_rate_limited()
        assert throttle.current_limit == 1

    def test_sustained_success_recovers_limit_up_to_max(self) -> None:
        from arandu.utils.concurrency import AdaptiveThrottle

        throttle = AdaptiveThrottle(4, cooldown_seconds=0.0, recovery_successes=2)
        throttle.acquire()
        throttle.release_rate_limited()  # 4 -> 2
        for _ in range(4):  # two batches of recovery_successes
            throttle.acquire()
            throttle.release_success()
        assert throttle.current_limit == 4
        for _ in range(10):
            throttle.acquire()
            throttle.release_success()
        assert throttle.current_limit == 4  # never exceeds max

    def test_plain_failure_neither_shrinks_nor_recovers(self) -> None:
        from arandu.utils.concurrency import AdaptiveThrottle

        throttle = AdaptiveThrottle(4, cooldown_seconds=0.0, recovery_successes=1)
        throttle.acquire()
        throttle.release_rate_limited()  # 4 -> 2
        throttle.acquire()
        throttle.release_failure()
        assert throttle.current_limit == 2

    def test_cooldown_blocks_acquire(self) -> None:
        import time

        from arandu.utils.concurrency import AdaptiveThrottle

        throttle = AdaptiveThrottle(2, cooldown_seconds=0.2)
        throttle.acquire()
        throttle.release_rate_limited()
        start = time.monotonic()
        throttle.acquire()
        waited = time.monotonic() - start
        throttle.release_success()
        assert waited >= 0.15


class TestMapConcurrentRateLimits:
    def test_rate_limited_item_is_requeued_and_succeeds(self) -> None:
        attempts: dict[int, int] = {}

        def flaky(x: int) -> int:
            attempts[x] = attempts.get(x, 0) + 1
            if x == 2 and attempts[x] == 1:
                raise FakeRateLimit("quota")
            return x * 10

        triples = list(
            map_concurrent(
                flaky,
                [1, 2, 3],
                workers=2,
                rate_limit_of=_is_fake_rate_limit,
                cooldown_seconds=0.0,
            )
        )

        assert sorted(t[1] for t in triples) == [10, 20, 30]
        assert all(t[2] is None for t in triples)
        assert attempts[2] == 2

    def test_persistent_rate_limit_fails_after_requeue_cap(self) -> None:
        calls: dict[int, int] = {}

        def always_limited(x: int) -> int:
            calls[x] = calls.get(x, 0) + 1
            raise FakeRateLimit("quota")

        triples = list(
            map_concurrent(
                always_limited,
                [7],
                workers=2,
                rate_limit_of=_is_fake_rate_limit,
                cooldown_seconds=0.0,
                max_rate_limit_requeues=3,
            )
        )

        assert len(triples) == 1
        assert isinstance(triples[0][2], FakeRateLimit)
        assert calls[7] == 4  # initial attempt + 3 requeues

    def test_sequential_mode_also_requeues_rate_limits(self) -> None:
        attempts: dict[int, int] = {}

        def flaky(x: int) -> int:
            attempts[x] = attempts.get(x, 0) + 1
            if attempts[x] == 1:
                raise FakeRateLimit("quota")
            return x

        triples = list(
            map_concurrent(
                flaky,
                [5],
                workers=1,
                rate_limit_of=_is_fake_rate_limit,
                cooldown_seconds=0.0,
            )
        )

        assert triples[0][1] == 5
        assert triples[0][2] is None
        assert attempts[5] == 2

    def test_non_rate_limit_errors_are_not_requeued(self) -> None:
        calls: dict[int, int] = {}

        def broken(x: int) -> int:
            calls[x] = calls.get(x, 0) + 1
            raise ValueError("not a quota problem")

        triples = list(
            map_concurrent(
                broken,
                [1],
                workers=2,
                rate_limit_of=_is_fake_rate_limit,
                cooldown_seconds=0.0,
            )
        )

        assert isinstance(triples[0][2], ValueError)
        assert calls[1] == 1
