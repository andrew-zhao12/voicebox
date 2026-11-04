"""Actual tiny Transformers 5 cache tests; run in the optional Nemotron environment.

No weights are downloaded. The normal desktop/torch-free suite skips these tests.
"""

import importlib.util

import numpy as np
import pytest

from backend.backends.nemotron_worker import NativeNemotronSession


@pytest.fixture(params=[False, True], ids=["english", "multilingual"])
def native_model(request):
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("requires the isolated Transformers 5 Nemotron environment")
    import transformers

    if not hasattr(transformers, "NemotronAsrStreamingConfig"):
        pytest.skip("requires Transformers 5.13")
    import torch
    from tokenizers import Tokenizer, models
    from transformers import (
        AutoModelForRNNT,
        NemotronAsrStreamingConfig,
        NemotronAsrStreamingEncoderConfig,
        NemotronAsrStreamingFeatureExtractor,
        NemotronAsrStreamingProcessor,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(13)
    torch.set_num_threads(1)
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
        tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, **{str(i): i for i in range(1, 8)}, "<blank>": 8})),
        pad_token="<pad>",
    )
    extractor = NemotronAsrStreamingFeatureExtractor(feature_size=16)
    if request.param:
        from transformers import Nemotron3_5AsrConfig, Nemotron3_5AsrProcessor

        config = Nemotron3_5AsrConfig(**params, num_prompts=2, prompt_intermediate_size=16, default_prompt_id=0)
        processor = Nemotron3_5AsrProcessor(
            extractor, tokenizer, prompt_dictionary={"auto": 0, "en-US": 1}, num_prompts=2
        )
    else:
        config = NemotronAsrStreamingConfig(**params)
        processor = NemotronAsrStreamingProcessor(extractor, tokenizer)
    return AutoModelForRNNT.from_config(config).eval(), processor, "en-US" if request.param else "en"


def test_native_cache_state_is_independent_and_packetization_invariant(native_model):
    model, processor, language = native_model
    whole = NativeNemotronSession(model, processor, language=language)
    packets = NativeNemotronSession(model, processor, language=language)
    audio = np.random.default_rng(42).normal(0, 0.01, 24017).astype(np.float32)
    whole.accept(audio, final=True)
    for offset in range(0, len(audio), 317):
        packets.accept(audio[offset : offset + 317])
        assert len(packets.buffer) < 10000
    final = packets.accept(np.empty(0, dtype=np.float32), final=True)
    assert whole.tokens == packets.tokens
    assert whole.token_frames == packets.token_frames
    assert whole.encoder_cache is not packets.encoder_cache
    assert whole.padding_cache is not packets.padding_cache
    assert whole.decoder_cache is not packets.decoder_cache
    assert packets.encoder_frames > 7
    assert final["duration"] == len(audio) / 16000
    assert all(segment["end"] <= final["duration"] for segment in final["segments"])


def test_short_final_chunk_and_empty_stream_finish(native_model):
    model, processor, language = native_model
    empty = NativeNemotronSession(model, processor, language=language)
    assert empty.accept(np.empty(0, dtype=np.float32), final=True)["text"] == ""
    short = NativeNemotronSession(model, processor, language=language)
    short.accept(np.zeros(17, dtype=np.float32))
    assert short.encoder_cache is None
    final = short.accept(np.empty(0, dtype=np.float32), final=True)
    assert short.encoder_cache is not None
    assert final["duration"] == 17 / 16000
    with pytest.raises(ValueError, match="already finished"):
        short.accept(np.zeros(1, dtype=np.float32))


def test_greedy_decoder_matches_upstream_generate(native_model, monkeypatch):
    model, processor, language = native_model
    session = NativeNemotronSession(model, processor, language=language)
    chunks = []
    prompt = {}
    encode = model.get_audio_features

    def record(features, **kwargs):
        chunks.append(features.clone())
        if "prompt_ids" in kwargs:
            prompt["prompt_ids"] = kwargs["prompt_ids"]
        return encode(features, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model, "get_audio_features", record)
        session.accept(np.random.default_rng(7).normal(0, 0.01, 18000).astype(np.float32), final=True)

    def features():
        yield from chunks

    reference = model.generate(
        input_features=features(),
        num_lookahead_tokens=session.right_context,
        return_dict_in_generate=True,
        decoder_start_token_id=model.config.blank_token_id,
        max_new_tokens=256,
        **prompt,
    )
    expected = [int(token) for token in reference.sequences[0] if token != model.config.blank_token_id]
    assert session.tokens == expected
