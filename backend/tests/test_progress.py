"""Model download progress: the ProgressManager, its SSE feed and the tqdm-based HuggingFace tracker."""

from __future__ import annotations

import asyncio
import json

from backend.utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from backend.utils.progress import ProgressManager, get_progress_manager

MB = 1_000_000


def test_progress_manager_stores_and_completes():
    pm = ProgressManager()
    pm.update_progress(model_name="test-model", current=50, total=100, filename="test.bin", status="downloading")
    progress = pm.get_progress("test-model")
    assert progress is not None
    assert progress["progress"] == 50.0
    assert progress["filename"] == "test.bin"
    assert progress["status"] == "downloading"

    pm.mark_complete("test-model")
    progress = pm.get_progress("test-model")
    assert progress["status"] == "complete"
    assert progress["progress"] == 100.0


async def _collect_until_terminal(pm: ProgressManager, model_name: str) -> list[dict]:
    events: list[dict] = []
    async for event in pm.subscribe(model_name):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            events.append(data)
            if data.get("status") in ("complete", "error"):
                break
    return events


async def test_progress_manager_streams_updates_over_sse():
    pm = ProgressManager()

    async def simulate_download():
        await asyncio.sleep(0.2)  # let the subscriber attach
        for i in range(0, 101, 20):
            pm.update_progress(
                model_name="sse-model", current=i, total=100, filename=f"file_{i}.bin", status="downloading"
            )
            await asyncio.sleep(0.05)
        pm.mark_complete("sse-model")

    events, _ = await asyncio.gather(_collect_until_terminal(pm, "sse-model"), simulate_download())
    assert events
    assert events[-1]["status"] == "complete"


def test_hf_progress_tracker_reports_byte_progress():
    """The tracker patches tqdm and reports once the known total passes 1 MB (small config files stay quiet)."""
    captured: list[tuple[int, int, str]] = []
    tracker = HFProgressTracker(lambda downloaded, total, filename: captured.append((downloaded, total, filename)))

    with tracker.patch_download():
        from tqdm import tqdm

        total_size = 5 * MB
        with tqdm(total=total_size, desc="model.bin", unit="B", unit_scale=True) as pbar:
            for _ in range(10):
                pbar.update(total_size // 10)

    assert captured, "no progress reported for a 5 MB download"
    last_downloaded = 0
    for downloaded, total, filename in captured:
        assert downloaded >= last_downloaded
        assert total == total_size
        assert filename == "model.bin"
        last_downloaded = downloaded
    assert captured[-1][0] == total_size

    quiet: list[tuple] = []
    small = HFProgressTracker(lambda *args: quiet.append(args))
    with small.patch_download():
        from tqdm import tqdm

        with tqdm(total=1000, desc="config.json", unit="B") as pbar:
            pbar.update(1000)
    assert quiet == []


async def test_tracker_and_manager_end_to_end():
    pm = get_progress_manager()
    model_name = "integration-test"

    async def simulate_real_download():
        await asyncio.sleep(0.2)
        tracker = HFProgressTracker(create_hf_progress_callback(model_name, pm))
        pm.update_progress(model_name=model_name, current=0, total=1, filename="", status="downloading")
        with tracker.patch_download():
            from tqdm import tqdm

            for filename, size in (("model.safetensors", 5 * MB), ("config.json", 1000)):
                with tqdm(total=size, desc=filename, unit="B") as pbar:
                    for chunk in range(0, size, size // 5):
                        pbar.update(min(size // 5, size - chunk))
                        await asyncio.sleep(0.01)
        pm.mark_complete(model_name)

    events, _ = await asyncio.gather(_collect_until_terminal(pm, model_name), simulate_real_download())
    assert events[-1]["status"] == "complete"
    assert any(event.get("filename") == "model.safetensors" for event in events)
