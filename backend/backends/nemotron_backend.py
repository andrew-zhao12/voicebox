"""Optional native Nemotron STT hosted in an isolated Transformers 5 interpreter."""

import asyncio
import base64
import os
from contextlib import suppress
from pathlib import Path

import numpy as np

from ..utils.subtitles import Transcript, segments_from_dicts
from .stt_capabilities import NEMOTRON_REPOS, nemotron_language

_WORKER = Path(__file__).with_name("nemotron_worker.py")
_INSTALL_MESSAGE = (
    "Nemotron runtime is unavailable. Install backend/requirements-nemotron.txt in a separate Python 3.12 environment "
    "and set VOICEBOX_NEMOTRON_PYTHON to its Python executable."
)


class NemotronSTTBackend:
    """Share weights across independent cached streams without changing desktop dependencies."""

    def __init__(self) -> None:
        self.model_size: str | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._sessions: set[str] = set()
        self._busy = False
        self._retiring: set[asyncio.Task] = set()

    @staticmethod
    def runtime_available() -> bool:
        """Check configured interpreter availability without importing or launching it."""
        python = os.environ.get("VOICEBOX_NEMOTRON_PYTHON", "")
        return bool(python and Path(python).is_file() and os.access(python, os.X_OK))

    @property
    def active_streams(self) -> int:
        """Count sessions pinning the currently loaded model."""
        return len(self._sessions)

    def is_loaded(self) -> bool:
        """Report whether the isolated model process is still alive."""
        return self.model_size is not None and self._process is not None and self._process.returncode is None

    def _is_model_cached(self, model_size: str) -> bool:
        from .base import is_model_cached

        repo = NEMOTRON_REPOS.get(model_size)
        required = [
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        return bool(repo and is_model_cached(repo, required_files=required))

    def _kill(self) -> asyncio.subprocess.Process | None:
        process = self._process
        self._process = None
        self.model_size = None
        self._sessions.clear()
        if process is not None:
            if process.stdin:
                process.stdin.close()
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
        return process

    def unload_model(self) -> None:
        """Stop the worker only when no stream or request still owns its model."""
        if self.active_streams or self._busy:
            raise RuntimeError("Nemotron has active streams or inference; close them before unloading its model")
        process = self._process
        self._process = None
        self.model_size = None
        if process is not None:
            process.stdin.close()
            task = asyncio.create_task(self._reap(process))
            self._retiring.add(task)
            task.add_done_callback(self._retiring.discard)

    async def _reap(self, process: asyncio.subprocess.Process) -> None:
        try:
            async with asyncio.timeout(5.0):
                await process.wait()
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def _rpc(self, request: dict, *, timeout: float = 120.0) -> dict:
        import json

        async with self._lock:
            process = self._process
            if process is None or process.returncode is not None:
                raise RuntimeError("Nemotron worker is unavailable; reopen the transcription session")
            self._busy = True
            try:
                async with asyncio.timeout(timeout):
                    process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
                    await process.stdin.drain()
                    line = await process.stdout.readline()
                    if not line:
                        raise BrokenPipeError("Nemotron worker exited before returning a result")
                    response = json.loads(line)
                if "error" in response:
                    raise RuntimeError(f"Nemotron inference failed: {response['error']}")
                return response["result"]
            except (
                asyncio.CancelledError,
                TimeoutError,
                BrokenPipeError,
                ConnectionResetError,
                json.JSONDecodeError,
            ) as exc:
                # Once a request is interrupted, a late reply cannot safely be
                # associated with another session. Killing also bounds cleanup.
                stopped = self._kill()
                if stopped is not None:
                    await asyncio.shield(stopped.wait())
                if isinstance(exc, (BrokenPipeError, ConnectionResetError, json.JSONDecodeError)):
                    raise RuntimeError("Nemotron worker disconnected; reopen the transcription session") from exc
                raise
            finally:
                self._busy = False

    async def _load(self, model_size: str) -> None:
        if model_size not in NEMOTRON_REPOS:
            raise ValueError(f"Unknown Nemotron model size: {model_size}")
        if self.is_loaded() and self.model_size == model_size:
            return
        if self.active_streams:
            raise RuntimeError("Nemotron has active streams; close them before switching models")
        if not self.runtime_available():
            raise RuntimeError(_INSTALL_MESSAGE)
        if self._process is not None:
            self.unload_model()
        if self._retiring:
            await asyncio.gather(*self._retiring)
        self._process = await asyncio.create_subprocess_exec(
            os.environ["VOICEBOX_NEMOTRON_PYTHON"],
            "-u",
            str(_WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )
        try:
            await self._rpc(
                {
                    "op": "load",
                    "repo": NEMOTRON_REPOS[model_size],
                    "device": os.environ.get("VOICEBOX_NEMOTRON_DEVICE", "auto"),
                },
                timeout=1800.0,
            )
        except (Exception, asyncio.CancelledError):
            stopped = self._kill()
            if stopped is not None:
                await asyncio.shield(stopped.wait())
            raise
        self.model_size = model_size

    async def load_model(self, model_size: str = "en-0.6b") -> None:
        """Load the selected model inside the optional runtime."""
        async with self._model_lock:
            await self._load(model_size)

    async def open_stream(
        self,
        language: str | None = None,
        model_size: str | None = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
        word_timestamps: bool = False,
    ) -> "NemotronStreamingSession":
        """Start independent cached decoding of mono float32 16 kHz audio."""
        from ..utils.speech_text import validate_hints

        validate_hints(prompt, hotwords)
        if prompt or hotwords:
            raise ValueError("Nemotron does not support text prompt or hotword hints")
        if word_timestamps:
            raise ValueError("Nemotron word timestamps are not supported; use faster-whisper for word alignment")
        async with self._model_lock:
            size = model_size or self.model_size or "en-0.6b"
            language = nemotron_language(language, size)
            await self._load(size)
            response = await self._rpc(
                {
                    "op": "open",
                    "language": language,
                    "delay_ms": int(os.environ.get("VOICEBOX_NEMOTRON_DELAY_MS", "560")),
                }
            )
            session_id = response["session"]
            self._sessions.add(session_id)
            return NemotronStreamingSession(self, session_id)

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
        word_timestamps: bool = False,
    ) -> str:
        """Transcribe a recording through the same cached decoder as realtime input."""
        result = await self.transcribe_detailed(
            audio_path, language, model_size, prompt=prompt, hotwords=hotwords, word_timestamps=word_timestamps
        )
        return result.text

    async def transcribe_detailed(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
        word_timestamps: bool = False,
    ) -> Transcript:
        """Return native emission-frame segment timing, without fabricated word alignment."""
        from ..utils.audio import load_audio

        session = await self.open_stream(
            language, model_size, prompt=prompt, hotwords=hotwords, word_timestamps=word_timestamps
        )
        try:
            audio, _ = await asyncio.to_thread(load_audio, audio_path, 16000)
            for start in range(0, len(audio), 16000):
                await session.accept_audio(audio[start : start + 16000])
            return await session.finish()
        finally:
            await session.close()


class NemotronStreamingSession:
    """An utterance's native caches, pinned until finish or explicit close."""

    def __init__(self, backend: NemotronSTTBackend, session_id: str) -> None:
        self._backend = backend
        self._id = session_id
        self._lock = asyncio.Lock()
        self._closed = False

    def _check_open(self) -> None:
        if self._closed or self._id not in self._backend._sessions:
            raise RuntimeError("Nemotron stream is closed or its worker was restarted")

    async def accept_audio(self, audio: np.ndarray) -> str:
        """Feed new audio once and return the cumulative transcript."""
        audio = np.asarray(audio, dtype="<f4")
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Nemotron requires finite mono float32 PCM at 16000 Hz")
        if len(audio) > 16000 * 30:
            raise ValueError("Each Nemotron audio chunk must be at most 30 seconds")
        async with self._lock:
            self._check_open()
            result = await self._backend._rpc(
                {"op": "accept", "session": self._id, "audio": base64.b64encode(audio.tobytes()).decode("ascii")}
            )
            return result["text"]

    async def finish(self) -> Transcript:
        """Flush the final partial native chunk and release the model pin."""
        async with self._lock:
            self._check_open()
            try:
                result = await self._backend._rpc({"op": "finish", "session": self._id})
                return Transcript(
                    text=result["text"],
                    segments=segments_from_dicts(result["segments"], duration=result["duration"]),
                    language=result["language"],
                    duration=result["duration"],
                )
            finally:
                self._closed = True
                self._backend._sessions.discard(self._id)

    async def close(self) -> None:
        """Discard caches without decoding buffered speech; safe after finish or cancellation."""
        async with self._lock:
            if self._closed:
                return
            try:
                if self._id in self._backend._sessions and self._backend.is_loaded():
                    await self._backend._rpc({"op": "close", "session": self._id}, timeout=5.0)
            finally:
                self._closed = True
                self._backend._sessions.discard(self._id)
