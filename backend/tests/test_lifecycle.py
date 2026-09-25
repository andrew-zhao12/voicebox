"""Draining, signal hooks, readiness and the queue shutdown (torch-free)."""

from __future__ import annotations

import asyncio
import os
import signal

import pytest
from fastapi.testclient import TestClient

from backend import lifecycle
from backend.services import task_queue
from backend.services.task_queue import QueueFullError, enqueue_generation, ensure_capacity, init_queue
from backend.tests.security_testapp import build_test_app


@pytest.fixture(autouse=True)
def _clean_lifecycle():
    lifecycle.reset()
    yield
    lifecycle.reset()


def test_begin_drain_is_idempotent_and_keeps_the_first_reason():
    assert not lifecycle.is_draining()
    assert lifecycle.begin_drain("test") is True
    assert lifecycle.begin_drain("later") is False
    assert lifecycle.is_draining()
    assert lifecycle.drain_reason() == "test"


def test_drain_timeout_env(caplog):
    assert lifecycle.drain_timeout_s({}) == lifecycle.DEFAULT_DRAIN_TIMEOUT_S
    assert lifecycle.drain_timeout_s({"VOICEBOX_DRAIN_TIMEOUT_S": "7.5"}) == 7.5
    assert lifecycle.drain_timeout_s({"VOICEBOX_DRAIN_TIMEOUT_S": "-3"}) == 0.0
    assert lifecycle.drain_timeout_s({"VOICEBOX_DRAIN_TIMEOUT_S": "soon"}) == lifecycle.DEFAULT_DRAIN_TIMEOUT_S


def test_signal_hook_wraps_the_existing_handler_and_restores_it():
    calls: list[int] = []

    def previous(signum, frame):
        calls.append(signum)

    original = signal.signal(signal.SIGUSR1, previous)
    try:
        assert lifecycle.install_signal_hooks((signal.SIGUSR1,)) == ["SIGUSR1"]
        # A second install is a no-op.
        assert lifecycle.install_signal_hooks((signal.SIGUSR1,)) == []
        os.kill(os.getpid(), signal.SIGUSR1)
        assert calls == [signal.SIGUSR1]
        assert lifecycle.is_draining()
        assert lifecycle.drain_reason() == "signal SIGUSR1"
        lifecycle.restore_signal_hooks()
        assert signal.getsignal(signal.SIGUSR1) is previous
    finally:
        signal.signal(signal.SIGUSR1, original)


def test_signals_without_a_python_handler_are_left_alone():
    original = signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    try:
        assert lifecycle.install_signal_hooks((signal.SIGUSR2,)) == []
        assert signal.getsignal(signal.SIGUSR2) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGUSR2, original)


async def test_wait_for_idle_returns_when_pending_drops_or_times_out():
    pending = [2]

    async def finish():
        await asyncio.sleep(0.05)
        pending[0] = 0

    asyncio.get_running_loop().create_task(finish())
    assert await lifecycle.wait_for_idle(lambda: pending[0], timeout_s=2.0, poll_s=0.01) is True
    pending[0] = 1
    assert await lifecycle.wait_for_idle(lambda: pending[0], timeout_s=0.05, poll_s=0.01) is False


def test_readiness_follows_preloads_worker_and_drain():
    ready, body = lifecycle.readiness(True)
    assert ready
    assert body["ready"]
    assert body["models"] == {"ready": [], "pending": {}, "failed": []}

    lifecycle.preload_expect(["kokoro", "whisper-turbo"])
    ready, body = lifecycle.readiness(True)
    assert not ready
    assert body["models"]["pending"] == {"kokoro": "queued", "whisper-turbo": "queued"}

    lifecycle.preload_phase("kokoro", "loading")
    lifecycle.preload_ready("kokoro")
    lifecycle.preload_failed("whisper-turbo", "HTTP 503 from the hub")
    ready, body = lifecycle.readiness(True)
    assert not ready
    assert body["models"]["ready"] == ["kokoro"]
    assert body["models"]["failed"] == ["whisper-turbo"]
    assert "503" not in str(body)  # error text stays in the log

    lifecycle.preload_ready("whisper-turbo")
    assert lifecycle.readiness(True)[0]
    assert not lifecycle.readiness(False)[0]

    lifecycle.begin_drain("test")
    ready, body = lifecycle.readiness(True)
    assert not ready
    assert body["draining"]


def test_readiness_route_is_public_and_reflects_draining(tmp_path):
    app, _runtime = build_test_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

        lifecycle.begin_drain("test")
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json() == {
            "ready": False,
            "draining": True,
            "worker": True,
            "models": {"ready": [], "pending": {}, "failed": []},
        }


async def test_queue_refuses_new_jobs_while_draining():
    init_queue(force=True)
    try:
        ensure_capacity()
        lifecycle.begin_drain("test")
        with pytest.raises(QueueFullError) as excinfo:
            ensure_capacity()
        assert excinfo.value.reason == "draining"
        assert excinfo.value.retry_after_s == 10
        assert "shutting down" in str(excinfo.value)

        async def never():
            pass

        with pytest.raises(QueueFullError):
            enqueue_generation("gen-1", never())
    finally:
        init_queue(force=True)
        await asyncio.sleep(0)


async def test_shutdown_waits_for_running_jobs_then_stops_the_worker():
    init_queue(force=True)
    finished = asyncio.Event()

    async def job():
        await asyncio.sleep(0.05)
        finished.set()

    enqueue_generation("gen-1", job())
    await asyncio.sleep(0)
    assert task_queue.worker_running()

    assert await task_queue.shutdown(drain_timeout_s=2.0) is True
    assert finished.is_set()
    assert task_queue.pending_count() == 0
    assert not task_queue.worker_running()
    assert lifecycle.is_draining()
    init_queue(force=True)
    await asyncio.sleep(0)


async def test_shutdown_cancels_running_jobs_after_the_drain_timeout():
    init_queue(force=True)
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def stuck():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def queued_later():
        raise AssertionError("must never run")

    enqueue_generation("gen-1", stuck())
    enqueue_generation("gen-2", queued_later())
    await started.wait()
    assert task_queue.pending_count() == 2

    assert await task_queue.shutdown(drain_timeout_s=0.05) is False
    assert cancelled.is_set()
    assert task_queue.pending_count() == 0
    assert not task_queue.worker_running()
    init_queue(force=True)
    await asyncio.sleep(0)
