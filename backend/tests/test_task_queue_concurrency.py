"""Same-engine concurrency within a lane (VOICEBOX_ENGINE_CONCURRENCY), torch-free."""

from __future__ import annotations

import asyncio

import pytest

from backend import lifecycle
from backend.observability import metrics
from backend.services import task_queue
from backend.services.task_queue import enqueue_generation, init_queue


@pytest.fixture(autouse=True)
async def _fresh_queue():
    lifecycle.reset()
    init_queue(force=True)
    yield
    init_queue(force=True)
    await asyncio.sleep(0)
    lifecycle.reset()


def test_configured_engine_concurrency_parsing(caplog):
    env = "VOICEBOX_ENGINE_CONCURRENCY"
    assert task_queue.configured_engine_concurrency({}) == {}
    assert task_queue.configured_engine_concurrency({env: "kokoro=2, Qwen=3"}) == {"kokoro": 2, "qwen": 3}
    assert task_queue.configured_engine_concurrency({env: "kokoro=1"}) == {}
    assert task_queue.configured_engine_concurrency({env: "kokoro=9"}) == {"kokoro": 4}
    assert task_queue.configured_engine_concurrency({env: "chatterbox=2"}) == {}
    assert "not reviewed for concurrent generation" in caplog.text
    assert task_queue.configured_engine_concurrency({env: "kokoro=lots,,=2"}) == {}
    assert "not engine=count" in caplog.text


async def blocked(started: asyncio.Event, release: asyncio.Event) -> None:
    started.set()
    await release.wait()


def _job():
    started, release = asyncio.Event(), asyncio.Event()
    return started, release, blocked(started, release)


async def _settled() -> None:
    for _ in range(6):
        await asyncio.sleep(0)


async def test_same_engine_jobs_overlap_up_to_the_limit():
    init_queue(force=True, engine_concurrency={"kokoro": 2})
    assert task_queue.engine_limit("kokoro") == 2
    assert task_queue.engine_limit("qwen") == 1
    assert task_queue.engine_limit(None) == 1
    s1, r1, c1 = _job()
    s2, r2, c2 = _job()
    s3, r3, c3 = _job()
    enqueue_generation("k1", c1, engine="kokoro")
    enqueue_generation("k2", c2, engine="kokoro")
    enqueue_generation("k3", c3, engine="kokoro")
    await s1.wait()
    await s2.wait()
    await _settled()
    assert not s3.is_set()  # two at once, the third waits
    assert len(task_queue._running_generation_tasks) == 2
    assert task_queue.engine_in_use("kokoro") == 3
    assert task_queue.pending_count() == 3

    r1.set()
    await s3.wait()  # a slot freed, the third starts while the second still runs
    assert not r2.is_set()
    r2.set()
    r3.set()
    await _settled()
    assert task_queue.pending_count() == 0
    assert task_queue._pending_by_owner == {}


async def test_a_different_engine_never_overtakes_and_waits_for_an_empty_lane():
    init_queue(force=True, engine_concurrency={"kokoro": 2})
    s1, r1, c1 = _job()
    s2, r2, c2 = _job()
    s3, r3, c3 = _job()
    enqueue_generation("k1", c1, engine="kokoro")
    enqueue_generation("q1", c2, engine="qwen")
    enqueue_generation("k2", c3, engine="kokoro")
    await s1.wait()
    await _settled()
    assert not s2.is_set()  # qwen waits for the lane to empty
    assert not s3.is_set()  # and kokoro #2 stays behind qwen (strict FIFO)

    r1.set()
    await s2.wait()
    await _settled()
    assert not s3.is_set()  # qwen runs alone (its limit is 1)
    r2.set()
    await s3.wait()
    r3.set()
    await _settled()
    assert task_queue.pending_count() == 0


async def test_exclusive_jobs_run_alone_within_a_lane():
    init_queue(force=True, engine_concurrency={"kokoro": 2})
    s1, r1, c1 = _job()
    s2, r2, c2 = _job()
    s3, r3, c3 = _job()
    enqueue_generation("k1", c1, engine="kokoro")
    enqueue_generation("seeded", c2, engine="kokoro", exclusive=True)
    enqueue_generation("k2", c3, engine="kokoro")
    await s1.wait()
    await _settled()
    assert not s2.is_set()  # the exclusive job waits for an empty lane
    r1.set()
    await s2.wait()
    await _settled()
    assert not s3.is_set()  # nothing overlaps an exclusive job
    r2.set()
    await s3.wait()
    r3.set()
    await _settled()


async def test_cancellation_of_running_queued_and_waiting_jobs():
    init_queue(force=True, engine_concurrency={"kokoro": 2})
    s1, r1, c1 = _job()
    s2, _r2, c2 = _job()
    s3, _r3, c3 = _job()
    s4, r4, c4 = _job()
    enqueue_generation("k1", c1, engine="kokoro")
    enqueue_generation("k2", c2, engine="kokoro")
    enqueue_generation("k3", c3, engine="kokoro")
    enqueue_generation("k4", c4, engine="kokoro")
    await s1.wait()
    await s2.wait()
    await _settled()

    assert task_queue.cancel_generation("k3") == "queued"  # head of the queue, waiting for a slot
    assert task_queue.cancel_generation("k2") == "running"
    await _settled()
    assert not s3.is_set()
    await s4.wait()  # k3 was skipped, k4 took the freed slot
    assert task_queue.engine_in_use("kokoro") == 2
    r1.set()
    r4.set()
    await _settled()
    assert task_queue.pending_count() == 0
    assert task_queue.cancel_generation("k1") is None


async def test_stopping_the_worker_cancels_every_running_job():
    init_queue(force=True, engine_concurrency={"kokoro": 2})
    s1, _r1, c1 = _job()
    s2, _r2, c2 = _job()
    enqueue_generation("k1", c1, engine="kokoro")
    enqueue_generation("k2", c2, engine="kokoro")
    await s1.wait()
    await s2.wait()
    tasks = list(task_queue._running_generation_tasks.values())
    worker = task_queue._generation_worker_task
    worker.cancel()
    await asyncio.wait_for(asyncio.gather(worker, *tasks, return_exceptions=True), timeout=1)
    assert worker.cancelled()
    assert all(task.cancelled() for task in tasks)


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_running_gauge_counts_concurrent_jobs():
    init_queue(force=True, engine_concurrency={"kokoro": 2})
    s1, r1, c1 = _job()
    s2, r2, c2 = _job()
    enqueue_generation("k1", c1, engine="kokoro")
    enqueue_generation("k2", c2, engine="kokoro")
    await s1.wait()
    await s2.wait()
    text = metrics.render()[0].decode()
    assert 'voicebox_queue_running_jobs{lane="all"} 2.0' in text
    r1.set()
    r2.set()
    await _settled()
    assert 'voicebox_queue_running_jobs{lane="all"} 0.0' in metrics.render()[0].decode()
