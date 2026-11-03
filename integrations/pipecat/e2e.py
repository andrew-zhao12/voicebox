"""Run bot.py with a scripted WebRTC caller: python e2e.py --stt realtime|batch.

Needs the integration requirements and a running Voicebox with Kokoro and Whisper.
No browser, Docker, microphone, or paid LLM is needed. The caller sends a synthesized
utterance over WebRTC and transcribes the bot's recorded greeting and echo reply.
Artifacts go to --out-dir (a new temporary directory by default).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

from bot import GREETING, Settings
from services import SAMPLE_RATE
from smoke import wav_bytes, word_error_rate

HERE = Path(__file__).resolve().parent
USER_LINE = "Hi there. Could you tell me what the weather is like today?"
FRAME_SAMPLES = SAMPLE_RATE // 50


class Microphone(AudioStreamTrack):
    """A paced microphone that sends silence until the caller starts its utterance."""

    def __init__(self, pcm: bytes):
        super().__init__()
        self.pcm = pcm
        self.offset = 0
        self.speaking = False
        self.done = asyncio.Event()
        self.pts = 0
        self.started: float | None = None
        self.speech_end = 0.0

    async def recv(self) -> AudioFrame:
        if self.started is None:
            self.started = time.monotonic()
        await asyncio.sleep(max(0, self.started + self.pts / SAMPLE_RATE - time.monotonic()))
        audio = b""
        if self.speaking and self.offset < len(self.pcm):
            audio = self.pcm[self.offset : self.offset + FRAME_SAMPLES * 2]
            self.offset += len(audio)
            if self.offset == len(self.pcm):
                self.speech_end = time.monotonic() + len(audio) / 2 / SAMPLE_RATE
                self.done.set()
        frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(audio.ljust(FRAME_SAMPLES * 2, b"\0"))
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self.pts
        frame.time_base = Fraction(1, SAMPLE_RATE)
        self.pts += FRAME_SAMPLES
        return frame


class Recording:
    """Collect received audio and detect audible starts and quiet tails."""

    def __init__(self):
        self.chunks: list[bytes] = []
        self.loud: list[tuple[int, float]] = []

    async def receive(self, track):
        resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        try:
            while True:
                frame = await track.recv()
                for mono in resampler.resample(frame):
                    pcm = mono.to_ndarray().reshape(-1)
                    if np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) > 500:
                        self.loud.append((len(self.chunks), time.monotonic()))
                    self.chunks.append(pcm.astype("<i2").tobytes())
        except MediaStreamError:
            return

    async def wait_quiet(self, after: int, timeout: float) -> float:
        """Require audible speech after *after*, then 1.5 seconds of silence."""
        async with asyncio.timeout(timeout):
            while not (
                self.loud
                and self.loud[-1][0] >= after
                and time.monotonic() - self.loud[-1][1] >= 1.5
            ):
                await asyncio.sleep(0.02)
        return next(at for index, at in self.loud if index >= after)

    def save(self, path: Path, start: int, end: int | None = None):
        path.write_bytes(wav_bytes(b"".join(self.chunks[start:end])))


async def keep_alive(channel):
    """SmallWebRTC uses application pings to detect a disconnected caller."""
    while True:
        if channel.readyState == "open":
            channel.send(f"ping {time.time()}")
        await asyncio.sleep(1)


async def conversation(url: str, pcm: bytes, out: Path, timeout: float) -> dict:
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    mic = Microphone(pcm)
    recording = Recording()
    tasks: list[asyncio.Task] = []
    peer.addTrack(mic)
    channel = peer.createDataChannel("chat")

    @peer.on("track")
    def on_track(track):
        if track.kind == "audio":
            tasks.append(asyncio.create_task(recording.receive(track)))

    try:
        await peer.setLocalDescription(await peer.createOffer())
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{url}/api/offer",
                json={"sdp": peer.localDescription.sdp, "type": peer.localDescription.type},
            ) as response:
                response.raise_for_status()
                answer = await response.json()
        await peer.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))
        tasks.append(asyncio.create_task(keep_alive(channel)))
        await recording.wait_quiet(0, timeout)
        greeting_end = len(recording.chunks)
        recording.save(out / "greeting.wav", 0, greeting_end)
        mic.speaking = True
        async with asyncio.timeout(timeout):
            await mic.done.wait()
        reply_start = await recording.wait_quiet(greeting_end, timeout)
        recording.save(out / "reply.wav", greeting_end)
        latency = reply_start - mic.speech_end
        if latency <= 0:
            raise RuntimeError("The bot replied before the caller finished speaking")
        return {"reply_latency_s": round(latency, 3)}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        mic.stop()
        await peer.close()


async def wait_ready(proc: subprocess.Popen, url: str, timeout: float):
    async with asyncio.timeout(timeout), aiohttp.ClientSession() as http:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"bot.py exited with status {proc.returncode}; see bot.log")
            try:
                async with http.get(f"{url}/openapi.json") as response:
                    if response.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.1)


def stop_worker(proc: subprocess.Popen):
    """Only stop the process group started by this check."""
    if proc.poll() is not None:
        return
    if hasattr(os, "killpg"):
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)


async def check(args) -> dict:
    settings = Settings.from_env(require_llm=False)
    out = Path(args.out_dir or tempfile.mkdtemp(prefix="voicebox-pipecat-e2e-")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {out}", flush=True)
    result = {"stt": args.stt, "pass": False}
    headers = {"Authorization": f"Bearer {settings.voicebox_key}"}
    proc = None
    try:
        async with aiohttp.ClientSession(
            headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as http:
            async with http.post(
                f"{settings.voicebox_url}/audio/speech",
                json={
                    "model": settings.tts_model,
                    "voice": "am_adam",
                    "input": USER_LINE,
                    "response_format": "pcm",
                },
            ) as response:
                response.raise_for_status()
                pcm = await response.read()
            if not pcm or len(pcm) % 2:
                raise RuntimeError("Voicebox returned invalid caller audio")
            (out / "user.wav").write_bytes(wav_bytes(pcm))
            # The runner cannot inherit a bound socket; choose an unused local port immediately
            # before launch and fail if the child cannot bind it.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            env = dict(
                os.environ,
                VOICEBOX_LLM="echo",
                VOICEBOX_STT_REALTIME="1" if args.stt == "realtime" else "0",
            )
            with (out / "bot.log").open("w") as log:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "bot.py",
                        "-t",
                        "webrtc",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    cwd=HERE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    await wait_ready(proc, url, args.timeout)
                    async with asyncio.timeout(args.timeout):
                        result.update(await conversation(url, pcm, out, args.timeout))
                finally:
                    await asyncio.to_thread(stop_worker, proc)
            for name, expected in [("greeting", GREETING), ("reply", f"You said: {USER_LINE}")]:
                form = aiohttp.FormData()
                form.add_field("model", settings.stt_model)
                form.add_field("language", settings.language)
                form.add_field(
                    "file",
                    (out / f"{name}.wav").read_bytes(),
                    filename=f"{name}.wav",
                    content_type="audio/wav",
                )
                async with http.post(
                    f"{settings.voicebox_url}/audio/transcriptions", data=form
                ) as response:
                    response.raise_for_status()
                    transcript = (await response.json())["text"].strip()
                wer = word_error_rate(expected, transcript)
                result[name] = {"transcript": transcript, "wer": round(wer, 3)}
                if wer > args.max_wer:
                    raise RuntimeError(f"{name} WER {wer:.2f} exceeds {args.max_wer:.2f}")
            if result["reply_latency_s"] > args.max_reply_latency:
                raise RuntimeError(f"Reply latency exceeds {args.max_reply_latency}s")
            result["pass"] = True
    except Exception as exc:
        result["error"] = str(exc) or type(exc).__name__
    finally:
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


async def main(args) -> int:
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    return 0 if (await check(args))["pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt", choices=["batch", "realtime"], default="realtime")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-wer", type=float, default=0.25)
    parser.add_argument("--max-reply-latency", type=float, default=5)
    parser.add_argument("--out-dir")
    try:
        sys.exit(asyncio.run(main(parser.parse_args())))
    except (KeyboardInterrupt, asyncio.CancelledError):
        sys.exit(130)
