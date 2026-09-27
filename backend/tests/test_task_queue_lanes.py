"""Two-lane scheduling: overlap across lanes, order within a lane, exclusive jobs, engine bookkeeping (torch-free)."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend import lifecycle
from backend.services import task_queue
from backend.services.task_queue import LANE_ALL, LANE_CPU, LANE_GPU, enqueue_generation, init_queue


@pytest.fixture(autouse=True)
async def _fresh_queue():
    lifecycle.reset()
    init_queue(force=True)
    yield
    init_queue(force=True)
    await asyncio.sleep(0)
    lifecycle.reset()


def test_configured_workers_parsing(caplog):
    assert task_queue.configured_workers({}) == 1
    assert task_queue.configured_workers({"VOICEBOX_GENERATION_WORKERS": "2"}) == 2
    assert task_queue.configured_workers({"VOICEBOX_GENERATION_WORKERS": "0"}) == 1
    assert task_queue.configured_workers({"VOICEBOX_GENERATION_WORKERS": "5"}) == 2
    assert task_queue.configured_workers({"VOICEBOX_GENERATION_WORKERS": "many"}) == 1
    assert "only the cpu and gpu lanes exist" in caplog.text


async def blocked(started: asyncio.Event, release: asyncio.Event) -> None:
    started.set()
    await release.wait()


def _job():
    started, release = asyncio.Event(), asyncio.Event()
    return started, release, blocked(started, release)


async def _settled() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_single_lane_keeps_everything_serial_and_ignores_lane_hints():
    init_queue(force=True, workers=1)
    assert task_queue.lane_names() == [LANE_ALL]
    assert not task_queue.parallel()
    assert task_queue.default_lane() == LANE_ALL
    assert task_queue.resolve_lane(LANE_CPU) == LANE_ALL

    s1, r1, c1 = _job()
    s2, r2, c2 = _job()
    enqueue_generation("g1", c1, lane=LANE_GPU)
    enqueue_generation("c1", c2, lane=LANE_CPU)
    await s1.wait()
    await _settled()
    assert not s2.is_set()
    r1.set()
    await asyncio.wait_for(s2.wait(), timeout=1)
    r2.set()
    await _settled()


async def test_two_lanes_overlap_but_serialize_within_a_lane():
    init_queue(force=True, workers=2)
    assert task_queue.lane_names() == [LANE_GPU, LANE_CPU]
    assert task_queue.parallel()
    assert task_queue.default_lane() == LANE_GPU
    assert task_queue.resolve_lane("bogus") == LANE_GPU
    assert task_queue.worker_running()

    gs1, gr1, g1 = _job()
    cs1, cr1, c1 = _job()
    gs2, gr2, g2 = _job()
    enqueue_generation("g1", g1, lane=LANE_GPU)
    enqueue_generation("c1", c1, lane=LANE_CPU)
    enqueue_generation("g2", g2, lane=LANE_GPU)
    await asyncio.wait_for(gs1.wait(), timeout=1)
    await asyncio.wait_for(cs1.wait(), timeout=1)  # the cpu job runs while the gpu job runs
    await _settled()
    assert not gs2.is_set()  # same lane: waits for g1
    assert task_queue.pending_count() == 3
    gr1.set()
    await asyncio.wait_for(gs2.wait(), timeout=1)
    cr1.set()
    gr2.set()
    await _settled()
    assert task_queue.pending_count() == 0


async def test_exclusive_job_runs_alone():
    init_queue(force=True, workers=2)
    gs1, gr1, g1 = _job()
    es1, er1, e1 = _job()
    cs2, cr2, c2 = _job()
    gs2, gr2, g2 = _job()
    enqueue_generation("g1", g1, lane=LANE_GPU)
    await asyncio.wait_for(gs1.wait(), timeout=1)
    enqueue_generation("e1", e1, lane=LANE_CPU, exclusive=True)
    enqueue_generation("c2", c2, lane=LANE_CPU)
    await _settled()
    assert not es1.is_set()  # exclusive: waits for the gpu lane to go idle
    assert not cs2.is_set()

    gr1.set()
    await asyncio.wait_for(es1.wait(), timeout=1)
    enqueue_generation("g2", g2, lane=LANE_GPU)
    await _settled()
    assert not gs2.is_set()  # nothing starts while the exclusive job runs
    assert not cs2.is_set()

    er1.set()
    await asyncio.wait_for(gs2.wait(), timeout=1)
    await asyncio.wait_for(cs2.wait(), timeout=1)
    gr2.set()
    cr2.set()
    await _settled()
    assert task_queue.pending_count() == 0


async def test_engine_bookkeeping_and_last_used():
    init_queue(force=True, workers=2)
    s1, r1, c1 = _job()
    _s2, _r2, c2 = _job()
    enqueue_generation("k1", c1, lane=LANE_CPU, engine="kokoro")
    enqueue_generation("k2", c2, lane=LANE_CPU, engine="kokoro")
    await asyncio.wait_for(s1.wait(), timeout=1)
    assert task_queue.engine_in_use("kokoro") == 2  # running plus queued
    assert task_queue.engine_in_use("qwen") == 0
    assert task_queue.cancel_generation("k2") == "queued"
    assert task_queue.engine_in_use("kokoro") == 1
    before = time.monotonic()
    r1.set()
    await _settled()
    assert task_queue.engine_in_use("kokoro") == 0
    assert task_queue.engines_last_used()["kokoro"] >= before
    c2.close()


async def test_shutdown_stops_every_lane():
    init_queue(force=True, workers=2)
    s1, _r1, c1 = _job()
    _s2, _r2, c2 = _job()
    enqueue_generation("g1", c1, lane=LANE_GPU)
    enqueue_generation("c1", c2, lane=LANE_CPU)
    await asyncio.wait_for(s1.wait(), timeout=1)
    assert await task_queue.shutdown(drain_timeout_s=0.05) is False
    assert not task_queue.worker_running()
    assert task_queue.pending_count() == 0
