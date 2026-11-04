"""Offline Linux image checks with real libraries and tiny random model weights.

Invoked by scripts/docker-stt-smoke.sh inside the selected runtime interpreter.
No checkpoint is downloaded and no recognition-quality claim is made.
"""

import argparse
import tempfile
import wave
from pathlib import Path


def faster_whisper() -> None:
    """Exercise PyAV decoding and the actual CTranslate2 CPU runtime."""
    import ctranslate2
    import numpy as np
    from faster_whisper.audio import decode_audio

    pcm = (np.sin(np.arange(1600) * 0.1) * 3000).astype("<i2").tobytes()
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
        with wave.open(audio.name, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        decoded = decode_audio(audio.name, sampling_rate=16000)
    if decoded.shape != (1600,) or not np.isfinite(decoded).all():
        raise RuntimeError("faster-whisper/PyAV failed to decode a PCM fixture")
    if "int8" not in ctranslate2.get_supported_compute_types("cpu"):
        raise RuntimeError("CTranslate2 does not offer the configured CPU INT8 runtime")
    print("faster-whisper: PCM decoder and CPU INT8 runtime passed")


def nemotron() -> None:
    """Run actual English/multilingual cached inference in the isolated runtime."""
    import importlib.util

    import numpy as np
    import torch
    from tokenizers import Tokenizer, models
    from transformers import (
        AutoModelForRNNT,
        Nemotron3_5AsrConfig,
        Nemotron3_5AsrProcessor,
        NemotronAsrStreamingConfig,
        NemotronAsrStreamingEncoderConfig,
        NemotronAsrStreamingFeatureExtractor,
        NemotronAsrStreamingProcessor,
        PreTrainedTokenizerFast,
    )

    path = Path(__file__).resolve().parents[1] / "backends" / "nemotron_worker.py"
    spec = importlib.util.spec_from_file_location("voicebox_nemotron_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.manual_seed(13)
    torch.set_num_threads(1)
    for multilingual in (False, True):
        encoder = NemotronAsrStreamingEncoderConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            subsampling_conv_channels=4,
            num_mel_bins=16,
        )
        params = dict(
            encoder_config=encoder,
            vocab_size=9,
            blank_token_id=8,
            decoder_hidden_size=16,
            num_decoder_layers=1,
            max_symbols_per_step=3,
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                models.WordLevel({"<pad>": 0, **{str(i): i for i in range(1, 8)}, "<blank>": 8})
            ),
            pad_token="<pad>",
        )
        extractor = NemotronAsrStreamingFeatureExtractor(feature_size=16)
        if multilingual:
            config = Nemotron3_5AsrConfig(**params, num_prompts=2, prompt_intermediate_size=16, default_prompt_id=0)
            processor = Nemotron3_5AsrProcessor(
                extractor, tokenizer, prompt_dictionary={"auto": 0, "en-US": 1}, num_prompts=2
            )
        else:
            config = NemotronAsrStreamingConfig(**params)
            processor = NemotronAsrStreamingProcessor(extractor, tokenizer)
        model = AutoModelForRNNT.from_config(config).eval()
        language = "en-US" if multilingual else "en"
        whole = worker.NativeNemotronSession(model, processor, language=language)
        packets = worker.NativeNemotronSession(model, processor, language=language)
        audio = np.random.default_rng(42).normal(0, 0.01, 24017).astype(np.float32)
        whole.accept(audio, final=True)
        for offset in range(0, len(audio), 317):
            packets.accept(audio[offset : offset + 317])
        final = packets.accept(np.empty(0, dtype=np.float32), final=True)
        if whole.tokens != packets.tokens or whole.encoder_cache is packets.encoder_cache:
            raise RuntimeError("Native packetization or independent cache smoke failed")
        if final["duration"] != len(audio) / 16000 or len(packets.buffer) >= 10000:
            raise RuntimeError("Native audio duration/buffer smoke failed")
    print("Nemotron: English and multilingual cached CPU inference passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("faster-whisper", "nemotron"))
    args = parser.parse_args()
    if args.runtime == "faster-whisper":
        faster_whisper()
    else:
        nemotron()


if __name__ == "__main__":
    main()
