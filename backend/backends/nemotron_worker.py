"""Isolated Transformers 5 worker with per-session cache-aware Nemotron decoding.

Executed as a script by nemotron_backend so the server's Transformers 4 package
and desktop TTS dependencies never enter this interpreter. Stdout is JSON IPC;
runtime diagnostics and model download progress belong on stderr.
"""

import base64
import copy
import json
import os
import sys
import uuid

import numpy as np


class NativeNemotronSession:
    """Incremental PCM frontend, encoder caches, and greedy RNNT state for one turn."""

    def __init__(self, model, processor, *, language: str | None, delay_ms: int = 560) -> None:
        import torch  # lazy: heavy import

        self.model = model
        self.processor = copy.copy(processor)
        supported = self.processor.supported_streaming_latencies_ms
        contexts = [right for right, delay in supported.items() if delay == delay_ms]
        if not contexts:
            raise ValueError(f"Unsupported Nemotron delay {delay_ms} ms; supported: {list(supported.values())}")
        self.right_context = contexts[0]
        self.processor.set_num_lookahead_tokens(self.right_context)
        self.language = language
        self.multilingual = model.config.model_type == "nemotron3_5_asr"
        if self.multilingual:
            from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
                Nemotron3_5AsrRNNTDecoderCache as DecoderCache,
            )
        else:
            from transformers.models.nemotron_asr_streaming.generation_nemotron_asr_streaming import (
                NemotronAsrStreamingRNNTDecoderCache as DecoderCache,
            )
        self.decoder_cache = DecoderCache(model.config)
        self.encoder_cache = None
        self.padding_cache = None
        self.last_token = torch.tensor([[model.config.blank_token_id]], device=model.device)
        self.tokens: list[int] = []
        self.token_frames: list[int] = []
        self.encoder_frames = 0
        self.mel_frames = 0
        self.samples = 0
        self.buffer_start = 0
        self.buffer = np.empty(0, dtype=np.float32)
        self.finished = False

    def _decode(self, audio: np.ndarray, *, first: bool, frames: int) -> None:
        import torch  # lazy: heavy import

        kwargs = {"language": self.language or "auto"} if self.multilingual else {}
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
            **kwargs,
        )
        features = inputs.input_features[:, :frames, :].to(self.model.device, dtype=self.model.dtype)
        if features.shape[1] != frames:
            raise RuntimeError(f"Nemotron frontend returned {features.shape[1]} mel frames; expected {frames}")
        kwargs = {"prompt_ids": inputs.prompt_ids.to(self.model.device)} if self.multilingual else {}
        with torch.inference_mode():
            output = self.model.get_audio_features(
                features,
                past_key_values=self.encoder_cache,
                padding_cache=self.padding_cache,
                num_lookahead_tokens=self.right_context,
                use_cache=True,
                output_attention_mask=False,
                **kwargs,
            )
            self.encoder_cache = output.past_key_values
            self.padding_cache = output.padding_cache
            for index in range(output.pooler_output.shape[1]):
                frame = output.pooler_output[:, index : index + 1, None, :]
                for _ in range(self.model.config.max_symbols_per_step):
                    decoder = self.model.decoder(self.last_token, cache=self.decoder_cache)
                    logits = self.model.joint(
                        encoder_hidden_states=frame,
                        decoder_hidden_states=decoder[:, None, :, :],
                    )
                    token = int(logits.reshape(-1).argmax())
                    self.last_token = torch.tensor([[token]], device=self.model.device)
                    if token == self.model.config.blank_token_id:
                        break
                    self.tokens.append(token)
                    self.token_frames.append(self.encoder_frames + index)
            self.encoder_frames += output.pooler_output.shape[1]

    def accept(self, audio: np.ndarray, *, final: bool = False) -> dict:
        """Process only new PCM, retaining bounded STFT overlap and native model caches."""
        if self.finished:
            raise ValueError("Nemotron session has already finished")
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Nemotron requires finite mono float32 PCM at 16000 Hz")
        self.samples += len(audio)
        self.buffer = np.concatenate((self.buffer, audio))
        fe = self.processor.feature_extractor
        while self.mel_frames * fe.hop_length < self.samples:
            first = self.mel_frames == 0
            frames = (
                self.processor.num_mel_frames_first_audio_chunk
                if first
                else self.processor.num_mel_frames_per_audio_chunk
            )
            start = 0 if first else self.mel_frames * fe.hop_length - fe.n_fft // 2
            # Include each frame's whole STFT window; padding a partial final
            # chunk is necessary because the cached encoder requires exact sizes.
            length = (frames - 1) * fe.hop_length + (fe.n_fft // 2 if first else fe.n_fft)
            end = start + length
            if end > self.samples and not final:
                break
            chunk = self.buffer[start - self.buffer_start : min(end, self.samples) - self.buffer_start]
            if len(chunk) < length:
                chunk = np.pad(chunk, (0, length - len(chunk)))
            self._decode(chunk, first=first, frames=frames)
            self.mel_frames += frames
            retain_from = max(0, self.mel_frames * fe.hop_length - fe.n_fft // 2)
            discard = min(len(self.buffer), retain_from - self.buffer_start)
            self.buffer = self.buffer[discard:]
            self.buffer_start += discard
        self.finished = final
        text = self.processor.decode(self.tokens, skip_special_tokens=True).strip() if self.tokens else ""
        # The checkpoint tokenizer marks multilingual locale tags as special,
        # so skip_special_tokens removes them without stripping spoken content.
        duration = self.samples / 16000
        start = min(duration, self.token_frames[0] * 0.08) if self.token_frames else 0.0
        end = min(duration, (self.token_frames[-1] + 1) * 0.08) if self.token_frames else 0.0
        return {
            "text": text,
            "language": self.language,
            "duration": duration,
            "segments": [{"id": 0, "start": start, "end": end, "text": text}] if text else [],
        }


class Runtime:
    """One shared model and independent stream states behind serialized IPC."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.sessions: dict[str, NativeNemotronSession] = {}

    def handle(self, request: dict) -> dict:
        """Dispatch a validated request from the parent server process."""
        operation = request["op"]
        if operation == "load":
            import torch  # lazy: heavy import
            from transformers import AutoModelForRNNT, AutoProcessor  # lazy: heavy import

            device = request["device"]
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device not in ("cpu", "cuda"):
                raise ValueError("VOICEBOX_NEMOTRON_DEVICE must be auto, cpu, or cuda")
            torch.set_num_threads(int(os.environ.get("VOICEBOX_REPLICA_THREADS", "4")))
            self.processor = AutoProcessor.from_pretrained(request["repo"])
            self.model = AutoModelForRNNT.from_pretrained(request["repo"], dtype=torch.float32).to(device).eval()
            return {"device": device}
        if operation == "open":
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = NativeNemotronSession(
                self.model, self.processor, language=request.get("language"), delay_ms=request["delay_ms"]
            )
            return {"session": session_id}
        session_id = request["session"]
        if operation == "close":
            self.sessions.pop(session_id, None)
            return {}
        session = self.sessions[session_id]
        if operation == "accept":
            audio = np.frombuffer(base64.b64decode(request["audio"], validate=True), dtype="<f4")
            return session.accept(audio)
        if operation == "finish":
            try:
                return session.accept(np.empty(0, dtype=np.float32), final=True)
            finally:
                self.sessions.pop(session_id, None)
        raise ValueError(f"Unknown Nemotron worker operation: {operation}")


def main() -> None:
    """Serve JSON lines until the parent closes stdin or terminates the worker."""
    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    runtime = Runtime()
    for line in sys.stdin:
        try:
            response = {"result": runtime.handle(json.loads(line))}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        protocol_output.write(json.dumps(response, allow_nan=False) + "\n")
        protocol_output.flush()


if __name__ == "__main__":
    main()
