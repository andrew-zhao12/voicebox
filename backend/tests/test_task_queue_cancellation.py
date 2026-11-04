import asyncio
import inspect

import pytest

from backend import lifecycle
from backend.services import task_queue


@pytest.mark.asyncio
async def test_cancel_queued_generation_skips_execution():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    release_running = asyncio.Event()
    queued_ran = asyncio.Event()

    async def running_job():
        running_started.set()
        await release_running.wait()

    async def queued_job():
        queued_ran.set()

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    task_queue.enqueue_generation("gen-queued", queued_job())
    assert task_queue.cancel_generation("gen-queued") == "queued"

    release_running.set()
    await asyncio.sleep(0.1)

    assert not queued_ran.is_set()


@pytest.mark.asyncio
async def test_cancel_running_generation_cancels_task():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    running_cancelled = asyncio.Event()

    async def running_job():
        running_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            running_cancelled.set()
            raise

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    assert task_queue.cancel_generation("gen-running") == "running"
    await asyncio.wait_for(running_cancelled.wait(), timeout=1)


@pytest.mark.parametrize(("workers", "exclusive"), [(1, False), (2, False), (2, True)])
async def test_cancel_before_first_step_releases_slots_and_starts_next_job(workers, exclusive, monkeypatch):
    lifecycle.reset()
    task_queue.init_queue(force=True, workers=workers)
    started, release = asyncio.Event(), asyncio.Event()
    next_started, next_release = asyncio.Event(), asyncio.Event()
    cpu_started, cpu_release = asyncio.Event(), asyncio.Event()
    releases = []
    original_release = task_queue._release

    def record_release(generation_id):
        releases.append(generation_id)
        original_release(generation_id)

    monkeypatch.setattr(task_queue, "_release", record_release)

    async def blocked(entered, released):
        entered.set()
        await released.wait()

    coro = blocked(started, release)
    task_queue.enqueue_generation("prestart", coro, owner="caller", engine="qwen", exclusive=exclusive)
    task_queue.enqueue_generation(
        "next", blocked(next_started, next_release), owner="caller", engine="kokoro", exclusive=True
    )
    try:
        await asyncio.sleep(0)  # dispatch happens before the new job task's first step
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
        assert task_queue.cancel_generation("prestart") == "running"
        await asyncio.wait_for(next_started.wait(), timeout=1)
        assert not started.is_set()
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        assert task_queue.pending_count() == 1
        assert task_queue._pending_by_owner == {"caller": 1}
        assert task_queue.engine_in_use("qwen") == 0
        assert task_queue.engine_in_use("kokoro") == 1
        assert releases.count("prestart") == 1
        next_task = task_queue._running_generation_tasks["next"]

        if workers == 2:
            assert task_queue._gate._active == 1
            assert task_queue._gate._exclusive
            task_queue.enqueue_generation("cpu", blocked(cpu_started, cpu_release), lane=task_queue.LANE_CPU)
            await asyncio.sleep(0)
            assert not cpu_started.is_set()  # the surviving exclusive job still owns the gate

        next_release.set()
        await asyncio.wait_for(next_task, timeout=1)
        if workers == 2:
            await asyncio.wait_for(cpu_started.wait(), timeout=1)
            cpu_task = task_queue._running_generation_tasks["cpu"]
            cpu_release.set()
            await asyncio.wait_for(cpu_task, timeout=1)
            assert task_queue._gate._active == 0
            assert not task_queue._gate._exclusive
        assert task_queue.pending_count() == 0
        assert task_queue._pending_by_owner == {}
        assert task_queue._job_engine == {}
        assert releases.count("next") == 1
        assert all(not lane.running and lane.running_engine is None for lane in task_queue._lanes.values())
    finally:
        release.set()
        next_release.set()
        cpu_release.set()
        await task_queue.shutdown(drain_timeout_s=0)
        lifecycle.reset()


async def test_shutdown_awaits_prestart_cleanup_before_releasing_capacity():
    lifecycle.reset()
    task_queue.init_queue(force=True, workers=2)
    gate = task_queue._gate
    coro = asyncio.Event().wait()
    task_queue.enqueue_generation("prestart", coro, owner="caller", engine="qwen", exclusive=True)
    await asyncio.sleep(0)
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
    lane = task_queue._lanes[task_queue.LANE_GPU]
    await gate._condition.acquire()  # hold cleanup after cancellation, so shutdown must wait for it
    assert task_queue.cancel_generation("prestart") == "running"
    shutdown = asyncio.create_task(task_queue.shutdown(drain_timeout_s=0))
    try:
        for _ in range(5):
            await asyncio.sleep(0)
        assert lane.cleanups
        assert not shutdown.done()
    finally:
        gate._condition.release()
        await asyncio.wait_for(shutdown, timeout=1)
        lifecycle.reset()
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
    assert task_queue.pending_count() == 0
    assert task_queue._pending_by_owner == {}
    assert task_queue._job_owner == {}
    assert task_queue._job_engine == {}
    assert not lane.running
    assert not lane.running_exclusive
    assert lane.running_engine is None
    assert gate._active == 0
    assert not gate._exclusive


@pytest.mark.parametrize("held", [False, True])
async def test_old_cleanup_does_not_release_reused_id_or_new_gate_after_reinit(held):
    lifecycle.reset()
    task_queue.init_queue(force=True, workers=2)
    old_gate = task_queue._gate
    old_lane = task_queue._lanes[task_queue.LANE_GPU]
    old_coro = asyncio.Event().wait()
    if held:
        task_queue.enqueue_generation("old-running", asyncio.Event().wait(), exclusive=True)
    task_queue.enqueue_generation("reused", old_coro, owner="old", engine="qwen", exclusive=True)
    await asyncio.sleep(0)
    assert inspect.getcoroutinestate(old_coro) == inspect.CORO_CREATED
    await old_gate._condition.acquire()
    started, release = asyncio.Event(), asyncio.Event()

    async def replacement():
        started.set()
        await release.wait()

    task_queue.init_queue(force=True, workers=2)
    task_queue.enqueue_generation("reused", replacement(), owner="new", engine="kokoro")
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert old_lane.cleanups
    finally:
        old_gate._condition.release()
    try:
        await asyncio.wait_for(asyncio.gather(*old_lane.cleanups), timeout=1)
        assert inspect.getcoroutinestate(old_coro) == inspect.CORO_CLOSED
        assert old_gate._active == 0
        assert not old_gate._exclusive
        assert task_queue._gate is not old_gate
        assert task_queue._gate._active == 1
        assert task_queue.pending_count() == 1
        assert task_queue._pending_by_owner == {"new": 1}
        assert task_queue.engine_in_use("kokoro") == 1
        assert task_queue.engine_in_use("qwen") == 0
        task = task_queue._running_generation_tasks["reused"]
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, timeout=1)
        assert task_queue.pending_count() == 0
        assert task_queue._gate._active == 0
    finally:
        release.set()
        await task_queue.shutdown(drain_timeout_s=0)
        lifecycle.reset()
