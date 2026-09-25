"""Tests for the one-at-a-time inference slots."""

import asyncio

import pytest

from backend.services.inference_slots import InferenceBusyError, InferenceSlot


async def test_waiters_are_bounded_and_served_in_order():
    slot = InferenceSlot("test", max_waiters=1, wait_timeout_s=5)
    release = asyncio.Event()
    order: list[str] = []

    async def hold():
        async with slot.acquire():
            order.append("first")
            await release.wait()

    async def wait_then_run():
        async with slot.acquire():
            order.append("second")

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(wait_then_run())
    await asyncio.sleep(0)
    assert slot.busy

    with pytest.raises(InferenceBusyError) as excinfo:
        async with slot.acquire():
            pass
    assert excinfo.value.retry_after_s == 5

    release.set()
    await asyncio.gather(holder, waiter)
    assert order == ["first", "second"]
    assert not slot.busy


async def test_waiting_times_out():
    slot = InferenceSlot("test", max_waiters=4, wait_timeout_s=0.01)
    release = asyncio.Event()

    async def hold():
        async with slot.acquire():
            await release.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    with pytest.raises(InferenceBusyError):
        async with slot.acquire():
            pass
    release.set()
    await holder
