"""Tests for long-form transcription on the PyTorch Whisper path (no model needed)."""

import numpy as np
import pytest

from backend.backends import pytorch_backend
from backend.backends.pytorch_backend import WHISPER_WINDOW_SAMPLES, PyTorchSTTBackend


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, audio, **kwargs):
        self.calls.append(kwargs)
        inputs = FakeInputs(input_features="features")
        if kwargs.get("return_attention_mask"):
            inputs["attention_mask"] = "mask"
        return inputs

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" hello world "]


class FakeModel:
    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, input_features, **kwargs):
        self.calls.append(kwargs)
        return ["ids"]


@pytest.fixture
def backend(monkeypatch):
    backend = PyTorchSTTBackend.__new__(PyTorchSTTBackend)
    backend.processor = FakeProcessor()
    backend.model = FakeModel()
    backend.model_size = "base"
    backend.device = "cpu"

    async def no_load(model_size=None):
        return None

    monkeypatch.setattr(backend, "load_model_async", no_load)
    return backend


def fake_audio(seconds):
    def load(path, sample_rate=16000):
        return np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate

    return load


async def test_short_audio_uses_the_padded_single_window(backend, monkeypatch):
    monkeypatch.setattr(pytorch_backend, "load_audio", fake_audio(10))

    text = await backend.transcribe("clip.wav", language="en")

    assert text == "hello world"
    assert backend.processor.calls == [{"sampling_rate": 16000, "return_tensors": "pt"}]
    assert backend.model.calls == [{"language": "en", "task": "transcribe"}]


async def test_long_audio_enables_long_form_generation(backend, monkeypatch):
    monkeypatch.setattr(pytorch_backend, "load_audio", fake_audio(61))

    await backend.transcribe("clip.wav")

    kwargs = backend.processor.calls[0]
    assert kwargs["truncation"] is False
    assert kwargs["padding"] == "longest"
    assert kwargs["return_attention_mask"] is True
    assert backend.model.calls == [{"attention_mask": "mask", "return_timestamps": True}]
    assert WHISPER_WINDOW_SAMPLES == 30 * 16000
