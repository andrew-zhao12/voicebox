"""Tests for the generation queue's global depth and per-owner caps."""

import asyncio

import pytest

from backend.services import task_queue


async def blocker(release: asyncio.Event):
    await release.wait()


async def noop():
    return None


async def test_owner_cap_and_global_depth():
    task_queue.init_queue(force=True, max_depth=3)
    release = asyncio.Event()

    task_queue.enqueue_generation("a1", blocker(release), owner="a", max_pending=2)
    await asyncio.sleep(0)  # let the worker start a1
    task_queue.enqueue_generation("a2", noop(), owner="a", max_pending=2)
    with pytest.raises(task_queue.QueueFullError) as owner_full:
        task_queue.enqueue_generation("a3", noop(), owner="a", max_pending=2)
    assert owner_full.value.reason == "owner"

    task_queue.enqueue_generation("b1", noop(), owner="b", max_pending=2)
    with pytest.raises(task_queue.QueueFullError) as global_full:
        task_queue.enqueue_generation("c1", noop(), owner="c")
    assert global_full.value.reason == "global"
    assert task_queue.pending_count() == 3

    release.set()
    await asyncio.sleep(0.05)
    assert task_queue.pending_count() == 0
    assert task_queue._pending_by_owner == {}


async def test_cancelling_a_queued_job_releases_the_owner_slot():
    task_queue.init_queue(force=True, max_depth=8)
    release = asyncio.Event()
    task_queue.enqueue_generation("a1", blocker(release), owner="a", max_pending=2)
    await asyncio.sleep(0)
    task_queue.enqueue_generation("a2", noop(), owner="a", max_pending=2)

    assert task_queue.cancel_generation("a2") == "queued"
    task_queue.enqueue_generation("a3", noop(), owner="a", max_pending=2)  # slot came back

    release.set()
    await asyncio.sleep(0.05)
    assert task_queue._pending_by_owner == {}


async def test_ensure_capacity_and_rejected_coroutines_are_closed():
    task_queue.init_queue(force=True, max_depth=1)
    release = asyncio.Event()
    task_queue.enqueue_generation("a1", blocker(release), owner="a")
    await asyncio.sleep(0)

    try:
        with pytest.raises(task_queue.QueueFullError):
            task_queue.ensure_capacity("b", None)
        coro = noop()
        with pytest.raises(task_queue.QueueFullError):
            task_queue.enqueue_generation("b1", coro, owner="b")
        with pytest.raises(RuntimeError, match="cannot reuse already awaited coroutine"):
            coro.send(None)
    finally:
        release.set()
        await asyncio.sleep(0.05)


async def test_worker_cancellation_wins_over_a_cancelled_job():
    task_queue.init_queue(force=True)
    release = asyncio.Event()
    task_queue.enqueue_generation("a1", blocker(release), owner="a")
    await asyncio.sleep(0)

    worker = task_queue._generation_worker_task
    job = task_queue._running_generation_tasks["a1"]
    job.cancel()
    worker.cancel()
    await asyncio.wait_for(asyncio.gather(worker, job, return_exceptions=True), timeout=1)
    assert worker.cancelled()


async def test_legacy_callers_without_an_owner_still_work():
    task_queue.init_queue(force=True)
    done = asyncio.Event()

    async def job():
        done.set()

    task_queue.enqueue_generation("x", job())
    await asyncio.wait_for(done.wait(), timeout=1)
