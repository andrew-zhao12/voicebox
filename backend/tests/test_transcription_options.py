"""STT vocabulary/timing options reach the selected model through the HTTP API."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.routes import transcription
from backend.services import transcribe
from backend.services.inference_slots import InferenceSlot
from backend.tests.test_openai_compat import api as shared_api, bearer, sample_wav
from backend.utils.subtitles import Transcript, TranscriptWord

api = shared_api


@pytest.mark.parametrize(("model", "size"), [("whisper-base", "base"), ("faster-whisper-small", "small")])
def test_model_options_reach_backend_and_words_are_returned(api, monkeypatch, model, size):
    calls = []

    async def detailed(path, language, model_size, **options):
        assert Path(path).exists()
        calls.append((language, model_size, options))
        return Transcript("Voicebox", language="en", duration=1, words=(TranscriptWord("Voicebox", 0.1, 0.8),))

    backend = SimpleNamespace(model_size=size, is_loaded=lambda: True, transcribe_detailed=detailed)
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: backend)
    monkeypatch.setattr("backend.backends.get_stt_backend_for_engine", lambda engine: backend)
    response = api.client.post(
        "/v1/audio/transcriptions",
        headers=bearer(api.key),
        files={"file": ("sample.wav", sample_wav(), "audio/wav")},
        data={
            "model": model,
            "language": "en",
            "prompt": "Voice software.",
            "hotwords": ["Voicebox", "Pipecat"],
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["words"] == [{"word": "Voicebox", "start": 0.1, "end": 0.8}]
    assert calls == [
        ("en", size, {"prompt": "Voice software.", "hotwords": ["Voicebox", "Pipecat"], "word_timestamps": True})
    ]


def test_native_stt_refuses_unsupported_hints_before_loading(api):
    response = api.client.post(
        "/v1/audio/transcriptions",
        headers=bearer(api.key),
        files={"file": ("sample.wav", sample_wav(), "audio/wav")},
        data={"model": "nemotron-streaming-en-0.6b", "prompt": "A hint."},
    )
    assert response.status_code == 400
    assert "does not support prompt" in response.json()["error"]["message"]


def test_word_timestamps_need_verbose_json(api):
    response = api.client.post(
        "/v1/audio/transcriptions",
        headers=bearer(api.key),
        files={"file": ("sample.wav", sample_wav(), "audio/wav")},
        data={"timestamp_granularities[]": "word"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


async def test_cancelled_upload_retains_its_owned_wav_until_inference_finishes(tmp_path, monkeypatch):
    original = tmp_path / "upload.wav"
    original.write_bytes(sample_wav())
    started, release = asyncio.Event(), asyncio.Event()
    paths = []

    async def detailed(path, *_args):
        paths.append(Path(path))
        started.set()
        await release.wait()
        assert Path(path).exists()
        return Transcript("hello")

    backend = SimpleNamespace(model_size="base", is_loaded=lambda: True, transcribe_detailed=detailed)
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: backend)
    slot = InferenceSlot("test")
    monkeypatch.setattr(transcription, "whisper_slot", slot)
    task = asyncio.create_task(transcription.transcribe_file_detailed(str(original), "en", "base"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    original.unlink()
    assert paths[0].exists()
    release.set()
    for _ in range(100):
        if not slot.busy:
            break
        await asyncio.sleep(0.001)
    assert not paths[0].exists()
    assert not slot.busy


async def test_cancel_before_inference_first_step_keeps_transferred_file(tmp_path, monkeypatch):
    original = tmp_path / "upload.wav"
    original.write_bytes(sample_wav())
    scheduled, release = asyncio.Event(), asyncio.Event()
    called = []

    class DeferredSlot:
        pending = None

        async def run(self, fn):
            # Match InferenceSlot's synchronous fn() handoff, but defer its
            # coroutine's first step to deterministically exercise cancellation.
            owned = fn()

            async def delayed():
                await release.wait()
                return await owned

            self.pending = asyncio.create_task(delayed())
            scheduled.set()
            return await asyncio.shield(self.pending)

    async def detailed(path, *_args):
        assert Path(path).exists()
        called.append(path)
        return Transcript("hello")

    backend = SimpleNamespace(model_size="base", is_loaded=lambda: True, transcribe_detailed=detailed)
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: backend)
    monkeypatch.setattr(transcription.tempfile, "tempdir", str(tmp_path))
    slot = DeferredSlot()
    monkeypatch.setattr(transcription, "whisper_slot", slot)
    request = asyncio.create_task(transcription.transcribe_file_detailed(str(original), "en", "base"))
    await asyncio.wait_for(scheduled.wait(), 5)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    try:
        assert not called
        assert len(list(tmp_path.glob("*.stt.wav"))) == 1
    finally:
        release.set()
        await slot.pending
    assert len(called) == 1
    assert not list(tmp_path.glob("*.stt.wav"))


async def test_upload_cancellation_removes_partially_written_temp(tmp_path, monkeypatch):
    waiting = asyncio.Event()

    class InterruptedUpload:
        filename = "recording.wav"
        reads = 0

        async def read(self, _size):
            self.reads += 1
            if self.reads == 1:
                return b"partial audio"
            waiting.set()
            await asyncio.Event().wait()
            return b""

    monkeypatch.setattr(transcription.tempfile, "tempdir", str(tmp_path))
    task = asyncio.create_task(transcription._save_upload(InterruptedUpload()))
    await asyncio.wait_for(waiting.wait(), 2)
    assert len(list(tmp_path.iterdir())) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(tmp_path.iterdir())
