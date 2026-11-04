"""Optional STT contracts, real subprocess lifecycle, and model metadata without ML weights."""

import asyncio
import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends import get_model_config, get_stt_backend_for_engine, get_stt_model_configs
from backend.backends.faster_whisper_backend import FasterWhisperSTTBackend
from backend.backends.nemotron_backend import NemotronSTTBackend
from backend.backends.stt_capabilities import nemotron_language


def test_registry_describes_actual_language_coverage_and_capabilities():
    configs = get_stt_model_configs()
    assert len(configs) == 12
    assert all(config.model_type == "stt" for config in configs)
    assert len(get_model_config("whisper-base").languages) == 99
    assert len(get_model_config("whisper-large").languages) == 100
    assert get_model_config("faster-whisper-base").supports_word_timestamps
    multilingual = get_model_config("nemotron-streaming-multilingual-0.6b")
    assert len(multilingual.languages) == 32
    assert "el-GR" not in multilingual.languages
    assert multilingual.supports_streaming
    assert not multilingual.supports_hotwords
    assert not multilingual.supports_word_timestamps
    assert get_stt_backend_for_engine("faster_whisper") is get_stt_backend_for_engine("faster_whisper")
    with pytest.raises(ValueError, match="Unknown STT engine"):
        get_stt_backend_for_engine("nonexistent")


@pytest.mark.parametrize(("hint", "expected"), [("en", "en-US"), ("fr", "fr-FR"), ("no", "nb-NO"), (None, None)])
def test_multilingual_aliases(hint, expected):
    assert nemotron_language(hint, "multilingual-0.6b") == expected


def test_language_errors_are_not_silently_ignored():
    with pytest.raises(ValueError, match="English only"):
        nemotron_language("es", "en-0.6b")
    with pytest.raises(ValueError, match="Unsupported"):
        nemotron_language("el", "multilingual-0.6b")


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 0.95


@dataclass
class Segment:
    id: int
    text: str
    start: float
    end: float
    words: tuple = ()
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@pytest.mark.asyncio
async def test_faster_whisper_consumes_lazy_decode_off_loop_and_keeps_aligned_words(monkeypatch):
    main_thread = threading.get_ident()
    options = {}

    class Model:
        def transcribe(self, path, **kwargs):
            assert threading.get_ident() != main_thread
            options.update(kwargs)

            def decode():
                assert threading.get_ident() != main_thread
                yield Segment(0, " hello", 0.0, 0.5, (Word(" hello", 0.1, 0.5),))
                yield Segment(1, " world", 0.5, 1.2, (Word(" world", 0.5, 1.2),))
                yield Segment(2, " hallucination", 2.0, 3.0, (Word(" hallucination", 2.0, 3.0),))

            return decode(), SimpleNamespace(language="en", duration=1.0)

    backend = FasterWhisperSTTBackend()
    backend.model = Model()
    backend.model_size = "base"
    result = await backend.transcribe_detailed(
        "test.wav", prompt="Context", hotwords=["Voicebox"], word_timestamps=True
    )
    assert result.text == "hello world"
    assert [word.word for word in result.words] == ["hello", "world"]
    assert result.words[-1].end == 1.0
    assert options == {"language": None, "initial_prompt": "Context", "hotwords": "Voicebox", "word_timestamps": True}


@pytest.mark.asyncio
async def test_faster_missing_optional_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace())
    with pytest.raises(RuntimeError, match=r"requirements-faster-whisper\.txt"):
        await FasterWhisperSTTBackend().load_model()


@pytest.fixture
def worker_backend(tmp_path, monkeypatch):
    """Use a real child process so cancellation and pipe ownership are exercised."""
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys, time\n"
        "sessions = {}\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line); op=r['op']; result={}\n"
        " if op == 'open':\n"
        "  key=str(len(sessions)+1); sessions[key]=0; result={'session':key}\n"
        " elif op == 'accept':\n"
        "  sessions[r['session']]+=1; result={'text':str(sessions[r['session']])}\n"
        " elif op == 'finish':\n"
        "  sessions.pop(r['session']); result={'text':'hello', 'segments':[], 'duration':1.0, 'language':'en'}\n"
        " elif op == 'close': sessions.pop(r['session'], None)\n"
        " elif op == 'hang': time.sleep(30)\n"
        " elif op == 'exit': sys.exit(3)\n"
        " print(json.dumps({'result':result}), flush=True)\n"
    )
    import backend.backends.nemotron_backend as module

    monkeypatch.setattr(module, "_WORKER", worker)
    monkeypatch.setenv("VOICEBOX_NEMOTRON_PYTHON", sys.executable)
    return NemotronSTTBackend()


async def retire(backend):
    backend.unload_model()
    if backend._retiring:
        await asyncio.gather(*backend._retiring)


@pytest.mark.asyncio
async def test_sessions_keep_independent_state_and_pin_models(worker_backend):
    backend = worker_backend
    first = await backend.open_stream("en")
    second = await backend.open_stream("en")
    assert backend.active_streams == 2
    assert await first.accept_audio(np.zeros(1600)) == "1"
    assert await first.accept_audio(np.zeros(1600)) == "2"
    assert await second.accept_audio(np.zeros(1600)) == "1"
    with pytest.raises(RuntimeError, match="active streams"):
        backend.unload_model()
    with pytest.raises(RuntimeError, match="active streams"):
        await backend.load_model("multilingual-0.6b")
    assert (await first.finish()).text == "hello"
    await first.close()
    await second.close()
    await second.close()
    assert backend.active_streams == 0
    process = backend._process
    await retire(backend)
    assert process.returncode == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_interrupted_ipc_terminates_worker_and_invalidates_sessions(worker_backend, cancel):
    backend = worker_backend
    session = await backend.open_stream()
    process = backend._process
    pending = asyncio.create_task(backend._rpc({"op": "hang"}, timeout=0.1 if not cancel else 30))
    await asyncio.sleep(0.03)
    if cancel:
        pending.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await asyncio.wait_for(pending, 2)
    assert process.returncode is not None
    assert not backend.is_loaded()
    assert backend.active_streams == 0
    await session.close()
    with pytest.raises(RuntimeError, match="closed"):
        await session.accept_audio(np.zeros(1600))
    replacement = await backend.open_stream()
    assert await replacement.accept_audio(np.zeros(1600)) == "1"
    await replacement.close()
    await retire(backend)


@pytest.mark.asyncio
async def test_unavailable_runtime_and_unsupported_hints_fail_before_spawn(monkeypatch):
    monkeypatch.delenv("VOICEBOX_NEMOTRON_PYTHON", raising=False)
    backend = NemotronSTTBackend()
    assert not backend.runtime_available()
    with pytest.raises(RuntimeError, match=r"separate Python 3\.12"):
        await backend.load_model()
    with pytest.raises(ValueError, match="hotword"):
        await backend.open_stream(hotwords=["Voicebox"])
    with pytest.raises(ValueError, match="word timestamps"):
        await backend.open_stream(word_timestamps=True)
    assert backend._process is None


@pytest.mark.asyncio
async def test_nonfinite_and_stereo_audio_rejected(worker_backend):
    session = await worker_backend.open_stream()
    for audio in [np.array([np.nan]), np.zeros((2, 100))]:
        with pytest.raises(ValueError, match="finite mono"):
            await session.accept_audio(audio)
    await session.close()
    await retire(worker_backend)


@pytest.mark.asyncio
async def test_worker_crash_does_not_leave_model_pinned(worker_backend):
    backend = worker_backend
    stale = await backend.open_stream()
    with pytest.raises(RuntimeError, match="disconnected"):
        await backend._rpc({"op": "exit"})
    assert backend.active_streams == 0
    assert not backend.is_loaded()
    await stale.close()
    replacement = await backend.open_stream()
    await replacement.close()
    await retire(backend)
