"""Scripted LiveKit participant: talks to the agent and records what it says.

Joins ROOM, publishes a microphone track that carries silence until it is
told to speak, and plays this script:

1. wait for the agent's greeting to finish;
2. say the --speak WAV and wait for the reply to start;
3. with --barge-in, say a second WAV --barge-after seconds into the reply,
   then wait for the reply to that;
4. wait for the agent to go quiet; repeat for --turns cycles, then leave.

It writes the agent's audio for each phase (greeting.wav, reply.wav and,
with --barge-in, barge-reply.wav) to --out-dir with a JSON report. Report
times are seconds since the room connected; add ``t0_wall`` for Unix time.

    participant.py --room R --speak user.wav --out-dir DIR
                   [--barge-in other.wav --barge-after 1.5] [--dispatch-agent NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import sys
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from livekit import api, rtc

REC_RATE = 48000
FRAME_S = 0.01


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """16-bit samples of a WAV, mixed to mono; rates that 10 ms frames can't split go to 48 kHz."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need a 16-bit PCM WAV")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate % 100:
        n = int(len(pcm) * REC_RATE / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.int16)
        rate = REC_RATE
    return pcm, rate


def speech_bounds(pcm: np.ndarray, rate: int, threshold: float) -> tuple[int, int]:
    """Indices of the first and last 10 ms frames above *threshold* RMS."""
    n = rate // 100
    frames = pcm[: len(pcm) // n * n].reshape(-1, n).astype(np.float32)
    loud = np.nonzero(np.sqrt((frames**2).mean(axis=1)) > threshold)[0]
    if loud.size == 0:
        raise SystemExit("the WAV to speak has no audio above --user-threshold")
    return int(loud[0]), int(loud[-1])


class Recorder:
    """The agent's audio frames, with their arrival times and loudness."""

    def __init__(self, t0: float, threshold: float) -> None:
        self.t0 = t0
        self.threshold = threshold
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []  # arrival of each frame's end
        self.loud: list[bool] = []
        self.loud_starts: list[float] = []  # start of each loud frame, ascending
        self.last_loud: float | None = None
        self.track_at: float | None = None

    def now(self) -> float:
        return time.monotonic() - self.t0

    def push(self, frame: rtc.AudioFrame) -> None:
        pcm = np.frombuffer(frame.data, dtype=np.int16).copy()
        t = self.now()
        loud = (
            bool(pcm.size) and float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) > self.threshold
        )
        self.frames.append(pcm)
        self.times.append(t)
        self.loud.append(loud)
        if loud:
            self.loud_starts.append(t - pcm.size / REC_RATE)
            self.last_loud = t

    def first_loud_after(self, t: float) -> float | None:
        """Start of the first loud frame that began at or after *t*."""
        i = bisect.bisect_left(self.loud_starts, t)
        return self.loud_starts[i] if i < len(self.loud_starts) else None

    def end_of_sound_after(self, t: float, quiet: float) -> float | None:
        """The agent's last loud frame after *t* that is followed by *quiet* seconds of silence.

        Returns *t* when the agent made no sound after *t*, and ``None`` while
        it has not been quiet for that long yet.
        """
        last = t
        for i in range(bisect.bisect_left(self.times, t), len(self.times)):
            if self.loud[i]:
                last = self.times[i]
            elif self.times[i] - last >= quiet:
                return last
        return last if self.now() - last >= quiet else None

    def loud_between(self, start: float, end: float) -> float:
        """Seconds of loud audio that arrived in [start, end)."""
        lo, hi = bisect.bisect_left(self.times, start), bisect.bisect_left(self.times, end)
        return sum(self.frames[i].size / REC_RATE for i in range(lo, hi) if self.loud[i])

    async def wait_for(self, predicate: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.02)
        return True

    async def wait_quiet(self, after: float, quiet: float, timeout: float) -> float | None:
        """Wait for sound after *after*, then *quiet* seconds of silence; returns the last sound."""
        ok = await self.wait_for(
            lambda: (
                self.last_loud is not None
                and self.last_loud >= after
                and self.now() - self.last_loud >= quiet
            ),
            timeout,
        )
        return self.last_loud if ok else None

    def write(self, path: Path, start: float, end: float) -> float:
        """Write the frames that arrived in [start, end) as a mono WAV; returns its duration."""
        lo, hi = bisect.bisect_left(self.times, start), bisect.bisect_left(self.times, end)
        pcm = np.concatenate(self.frames[lo:hi]) if hi > lo else np.zeros(0, np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(REC_RATE)
            w.writeframes(pcm.tobytes())
        return round(pcm.size / REC_RATE, 3)


@dataclass
class Mic:
    """A microphone track that always sends 10 ms frames: queued speech, else silence.

    A real microphone never stops, and the agent's VAD needs the trailing
    silence to notice that the user stopped talking.
    """

    room: rtc.Room
    rate: int
    t0: float
    queue: list[np.ndarray] = field(default_factory=list)
    sent_at: list[float] = field(default_factory=list)
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.n = self.rate // 100
        self.source = rtc.AudioSource(self.rate, 1)
        self.idle.set()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        track = rtc.LocalAudioTrack.create_audio_track("e2e-mic", self.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)
        self._task = asyncio.create_task(self._pump())

    async def say(self, pcm: np.ndarray, bounds: tuple[int, int]) -> tuple[float, float]:
        """Play *pcm* in real time; returns when its first and last loud frames were captured."""
        pcm = np.concatenate([pcm, np.zeros((-len(pcm)) % self.n, np.int16)])
        self.sent_at = []
        self.queue.extend(pcm.reshape(-1, self.n))
        self.idle.clear()
        await self.idle.wait()
        return self.sent_at[bounds[0]], self.sent_at[min(bounds[1], len(self.sent_at) - 1)]

    async def _pump(self) -> None:
        silence = np.zeros(self.n, np.int16)
        start, i = time.monotonic(), 0
        while True:
            delay = start + i * FRAME_S - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.2:  # fell behind (a busy machine): resync instead of bursting
                start, i = time.monotonic(), 0
            if self.queue:
                chunk = self.queue.pop(0)
                self.sent_at.append(time.monotonic() - self.t0)
                if not self.queue:
                    self.idle.set()
            else:
                chunk = silence
            await self.source.capture_frame(rtc.AudioFrame(chunk.tobytes(), self.rate, 1, self.n))
            i += 1

    async def aclose(self) -> None:
        if self._task:
            self._task.cancel()
        await self.source.aclose()


def join_token(identity: str, room: str) -> str:
    key, secret = (
        os.environ.get("LIVEKIT_API_KEY", "devkey"),
        os.environ.get("LIVEKIT_API_SECRET", "secret"),
    )
    grants = api.VideoGrants(room_join=True, room=room)
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .to_jwt()
    )


async def dispatch(url: str, agent_name: str, room: str) -> str:
    key, secret = (
        os.environ.get("LIVEKIT_API_KEY", "devkey"),
        os.environ.get("LIVEKIT_API_SECRET", "secret"),
    )
    async with api.LiveKitAPI(url.replace("ws", "http", 1), key, secret) as lk:
        req = api.CreateAgentDispatchRequest(agent_name=agent_name, room=room)
        return (await lk.agent_dispatch.create_dispatch(req)).id


async def run_turn(rec, mic, speech, barge, args) -> dict:
    """One normal/interrupted exchange on an already connected room."""
    phases = {}
    phases["user_start"], phases["user_end"] = await mic.say(
        speech[0], speech_bounds(*speech, args.user_threshold)
    )
    reply_seen = await rec.wait_for(
        lambda: rec.first_loud_after(phases["user_end"]) is not None, args.reply_timeout
    )
    if not reply_seen:
        raise RuntimeError("no reply after the user spoke")
    phases["reply_start"] = rec.first_loud_after(phases["user_end"])

    if barge:
        await asyncio.sleep(max(0.0, phases["reply_start"] + args.barge_after - rec.now()))
        if rec.last_loud is None or rec.now() - rec.last_loud > 0.2:
            raise RuntimeError("reply ended before barge-in; interruption was not exercised")
        bounds = speech_bounds(*barge, args.user_threshold)
        phases["barge_start"], phases["barge_end"] = await mic.say(barge[0], bounds)

        def stopped() -> bool:
            return rec.end_of_sound_after(phases["barge_start"], args.stop_gap) is not None

        if not await rec.wait_for(stopped, args.reply_timeout):
            raise RuntimeError("the agent kept talking over the barge-in")
        phases["agent_stop"] = rec.end_of_sound_after(phases["barge_start"], args.stop_gap)
        seen = await rec.wait_for(
            lambda: rec.first_loud_after(phases["barge_end"]) is not None, args.reply_timeout
        )
        if not seen:
            raise RuntimeError("no reply after the barge-in")
        phases["barge_reply_start"] = rec.first_loud_after(phases["barge_end"])
        end = await rec.wait_quiet(
            phases["barge_reply_start"], args.reply_silence, args.reply_timeout
        )
        if end is None:
            raise RuntimeError("reply after barge-in never ended")
        phases["barge_reply_end"] = end
    else:
        end = await rec.wait_quiet(phases["reply_start"], args.reply_silence, args.reply_timeout)
        if end is None:
            raise RuntimeError("reply never ended")
        phases["reply_end"] = end
    return phases


async def run(args: argparse.Namespace) -> dict:
    url = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    speech = load_wav_mono(args.speak)
    barge = load_wav_mono(args.barge_in) if args.barge_in else None
    if barge and barge[1] != speech[1]:
        raise SystemExit("--speak and --barge-in WAVs need the same sample rate")

    room = rtc.Room()
    t0 = time.monotonic()
    report: dict = {"room": args.room, "t0_wall": time.time() - (time.monotonic() - t0)}
    rec = Recorder(t0, args.rms_threshold)
    tasks: list[asyncio.Task] = []

    async def record(track: rtc.Track) -> None:
        rec.track_at = rec.now()
        async for ev in rtc.AudioStream(track, sample_rate=REC_RATE, num_channels=1):
            rec.push(ev.frame)

    @room.on("track_subscribed")
    def _on_track(
        track: rtc.Track, _pub: rtc.RemoteTrackPublication, _p: rtc.RemoteParticipant
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO and not tasks:  # the agent's voice
            tasks.append(asyncio.create_task(record(track)))

    await room.connect(url, join_token(args.identity, args.room))
    report["connect_s"] = round(rec.now(), 3)
    mic = Mic(room, speech[1], t0)
    await mic.start()
    if args.dispatch_agent:
        report["dispatch_id"] = await dispatch(url, args.dispatch_agent, args.room)

    phases: dict[str, float] = {}
    try:
        if not await rec.wait_for(lambda: rec.track_at is not None, args.join_timeout):
            report["error"] = "the agent published no audio track"
            return report
        report["agent_track_s"] = round(rec.track_at, 3)

        greeting_end = await rec.wait_quiet(0.0, args.greeting_gap, args.greeting_timeout)
        if greeting_end is None:
            report["error"] = "no greeting, or it never ended"
            return report
        phases["greeting_start"] = rec.first_loud_after(0.0) or 0.0
        phases["greeting_end"] = greeting_end

        report["turns"] = []
        rec.write(out / "greeting.wav", 0.0, rec.now())
        for number in range(1, args.turns + 1):
            if number > 1:
                await asyncio.sleep(args.idle_between)
            turn = await run_turn(rec, mic, speech, barge, args)
            prefix = "" if number == 1 else f"turn-{number}-"
            reply_end = turn.get("agent_stop", turn.get("reply_end", rec.now())) + FRAME_S
            rec.write(out / f"{prefix}reply.wav", turn["user_end"], reply_end)
            if barge:
                rec.write(out / f"{prefix}barge-reply.wav", turn["barge_end"], rec.now())
            evidence = {
                "turn": number,
                "phases": {k: round(v, 3) for k, v in turn.items()},
                "agent_audio_during_user_s": round(
                    rec.loud_between(turn["user_start"], turn["user_end"]), 3
                ),
            }
            if barge:
                evidence["agent_audio_during_barge_s"] = round(
                    rec.loud_between(turn["agent_stop"] + args.stop_gap, turn["barge_end"]), 3
                )
            report["turns"].append(evidence)
            if number == 1:
                phases.update(turn)
                report.update(
                    {
                        key: value
                        for key, value in evidence.items()
                        if key.startswith("agent_audio_")
                    }
                )
            # Keep long-call memory bounded; WAV artifacts retain completed turns.
            rec.frames.clear()
            rec.times.clear()
            rec.loud.clear()
            rec.loud_starts.clear()
            rec.last_loud = None
            (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    except RuntimeError as exc:
        report["error"] = str(exc)
    finally:
        await mic.aclose()
        for t in tasks:
            t.cancel()
        await room.disconnect()
        report["phases"] = {k: round(v, 3) for k, v in phases.items()}

    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--room", required=True)
    ap.add_argument("--speak", required=True, help="16-bit WAV the user says after the greeting")
    ap.add_argument("--out-dir", required=True, help="where the agent's audio and report.json go")
    ap.add_argument("--identity", default="e2e-user")
    ap.add_argument(
        "--dispatch-agent", help="ask the LiveKit server to dispatch this agent to the room"
    )
    ap.add_argument("--barge-in", help="16-bit WAV said over the agent's reply")
    ap.add_argument(
        "--barge-after", type=float, default=1.5, help="seconds into the reply to start talking"
    )
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--idle-between", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=120.0, help="hard cap for the whole session")
    ap.add_argument("--join-timeout", type=float, default=30.0)
    ap.add_argument("--greeting-timeout", type=float, default=30.0)
    ap.add_argument("--reply-timeout", type=float, default=30.0)
    ap.add_argument("--greeting-gap", type=float, default=1.2, help="quiet that ends the greeting")
    ap.add_argument("--reply-silence", type=float, default=2.0, help="quiet that ends a reply")
    ap.add_argument(
        "--stop-gap", type=float, default=1.0, help="quiet that counts as the agent stopping"
    )
    ap.add_argument(
        "--rms-threshold", type=float, default=500.0, help="loudness of agent audio (int16 RMS)"
    )
    ap.add_argument(
        "--user-threshold", type=float, default=300.0, help="loudness of the WAVs (int16 RMS)"
    )
    args = ap.parse_args()

    try:
        report = asyncio.run(asyncio.wait_for(run(args), timeout=args.timeout))
    except asyncio.TimeoutError:
        report = {"room": args.room, "error": f"session exceeded --timeout {args.timeout}s"}
    text = json.dumps(report, indent=2)
    (Path(args.out_dir) / "report.json").write_text(text + "\n")
    print(text)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
