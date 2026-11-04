"""End-to-end check: a LiveKit room, the example's voice pipeline and a real Voicebox.

    python run_e2e.py [--stt realtime|batch] [--sessions N] [--barge-in]
                      [--agent e2e|example] [--out-dir DIR]

Needs Docker, a running Voicebox with Kokoro and Whisper (VOICEBOX_URL) and a
client key (VOICEBOX_CLIENT_KEY). An admin key (VOICEBOX_ADMIN_KEY) lets the
barge-in check read Voicebox's /metrics. One run:

1. has Voicebox synthesize the user's lines in another voice (am_adam);
2. starts livekit-server in Docker (container lk-e2e on 7880, 7881 and 7882/udp);
3. starts the worker: e2e_agent.py (the example's session with an echo LLM),
   or with --agent example the unmodified ../agent.py, its LLM pointed at a
   local OpenAI-compatible echo server. It runs with LIVEKIT_DEV_MODE=1, where
   LiveKit's defaults would be its cloud turn detector and adaptive
   interruptions, and registers as e2e-agent-<run id>;
4. per session joins a new room with participant.py, has Voicebox transcribe
   what the agent said, checks it and collects timings;
5. stops everything, prints a summary and exits 0 when every session passed.
   SIGTERM, SIGHUP and SIGALRM stop a run the way Ctrl-C does, through the
   same teardown.

Each reply is "You said: <what the agent heard>", so its transcript checks
Voicebox's STT and TTS in one go. Artifacts (WAVs, reports, worker and
server logs) go to --out-dir, a new temporary directory by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent / "agent.py"
PY = sys.executable
AGENT_NAME = "e2e-agent"  # each run registers as e2e-agent-<run id>
USER_IDENTITY = "e2e-user"
USER_LINE = "Hi there. Could you tell me what the weather is like today?"
BARGE_LINE = "Sorry to interrupt. What time is it in Tokyo right now?"
# Appended to the first reply in --barge-in runs, so that speech is still being
# synthesized when the user cuts in. Each sentence is longer than Voicebox's
# first chunk (120 characters), so its request stays in flight while the rest
# renders; the barge-in lands inside a request rather than between two.
ECHO_TAIL = " ".join(
    f"This is part {n} of a deliberately long answer, which keeps going with plenty of words "
    "and clauses, so that the speech engine is still busy rendering it, sentence after "
    "sentence, at the moment you decide to cut in and ask me something else instead."
    for n in range(1, 61)
)
ABANDON_LOG_LINE = "abandoned before its first chunk"
# What the example logs for its session when turn-taking is pinned to local models.
LOCAL_TURN_HANDLING = "turn detection: turn-detector-v1-mini, interruptions: vad"
# kill, timeout, a closed terminal, an alarm: each ends a run through its teardown, as Ctrl-C does.
STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGALRM") if hasattr(signal, name)
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CONSOLE_RECORD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) - \w+ \S+ - (.*)$")


def log(msg: str) -> None:
    print(f"[e2e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " ")).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of *hypothesis* against *reference*, ignoring case and punctuation."""
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(len(ref), 1)


def example_greeting() -> str:
    """``GREETING`` from ../agent.py, read without importing (and starting) the example."""
    for node in ast.parse(EXAMPLE.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(getattr(t, "id", "") == "GREETING" for t in targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{EXAMPLE} defines no GREETING")


def port_busy(port: int, *, udp: bool = False) -> bool:
    """Whether something on this host already holds *port*.

    TCP is probed by connecting, not binding: after ``docker stop``, Docker's
    port proxy leaves TIME_WAIT sockets that make a bind() fail for about
    30 s although nothing listens. UDP has no TIME_WAIT, so a bind tells.
    """
    if udp:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError:
                return True
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def log_records(logfile: Path) -> Iterator[dict[str, Any]]:
    """The worker's log records: JSON lines (``start``) or console lines (``dev``, ``console``).

    Each comes back as its extra fields plus ``message`` and ``wall`` (Unix time).
    """
    for raw in logfile.read_text(errors="replace").splitlines():
        try:
            record = json.loads(raw)
        except ValueError:
            record = None
        if isinstance(record, dict) and "message" in record:
            stamp = record.get("timestamp")
            record["wall"] = datetime.fromisoformat(stamp).timestamp() if stamp else None
            yield record
            continue
        m = CONSOLE_RECORD.match(ANSI.sub("", raw))
        if not m:
            continue
        message, fields = m.group(3).rstrip(), {}
        # Console records end with their extra fields as JSON, which may nest.
        for start in (i for i, _ in enumerate(message) if message.startswith(" {", i)):
            try:
                fields = json.loads(message[start + 1 :])
            except ValueError:
                continue
            message = message[:start]
            break
        wall = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1e3
        yield {**fields, "message": message, "wall": wall}


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.3f} / {statistics.median(values):.3f} / {max(values):.3f}"


class Voicebox:
    """The Voicebox server under test, through the OpenAI SDK where it can be."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.url = args.voicebox_url.rstrip("/")
        self.args = args
        self.sdk = OpenAI(
            base_url=f"{self.url}/v1", api_key=args.client_key, max_retries=2, timeout=120
        )

    def _get(self, path: str, key: str | None) -> tuple[int, bytes]:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        req = urllib.request.Request(f"{self.url}{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except OSError:
            return 0, b""

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while (status := self._get("/health/ready", self.args.client_key)[0]) != 200:
            if time.monotonic() > deadline:
                raise SystemExit(f"{self.url}/health/ready answered {status or 'nothing'}")
            time.sleep(1)
        status, body = self._get("/v1", self.args.client_key)
        if status != 200:
            raise SystemExit(f"GET {self.url}/v1 with the client key: {status} {body[:200]!r}")

    def speak(self, text: str, voice: str, path: Path) -> float:
        """Synthesize *text* to a 16-bit WAV at *path*; returns its duration."""
        with self.sdk.audio.speech.with_streaming_response.create(
            model=self.args.tts_model, voice=voice, input=text, response_format="wav"
        ) as resp:
            raw = resp.read()
        # A streamed WAV has placeholder chunk sizes; rewrite it with real ones.
        fmt, data = raw.find(b"fmt "), raw.find(b"data")
        channels, rate = struct.unpack_from("<HI", raw, fmt + 10)
        pcm = raw[data + 8 :]
        with wave.open(str(path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        return len(pcm) / 2 / channels / rate

    def transcribe(self, path: Path) -> str:
        with path.open("rb") as f:
            result = self.sdk.audio.transcriptions.create(
                model=self.args.stt_model, file=f, language=self.args.language
            )
        return result.text.strip()

    def metrics(self) -> dict[str, float] | None:
        """The ``/metrics`` counters the checks use, or ``None`` without an admin key."""
        if not self.args.admin_key:
            return None
        status, body = self._get("/metrics", self.args.admin_key)
        if status != 200:
            log(f"GET /metrics answered {status}; Voicebox counters are left out")
            return None
        keys = (
            "speech_200",
            "speech_499",
            "transcriptions",
            "stream_cancelled",
            "stream_abandoned",
        )
        totals = dict.fromkeys(keys, 0.0)
        for line in body.decode().splitlines():
            m = re.match(r"^(\w+)\{(.*)\}\s+(\S+)$", line)
            if not m:
                continue
            name, value = m.group(1), float(m.group(3))
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
            route, status_label = labels.get("route"), labels.get("status")
            if name == "voicebox_http_requests_total":
                if route == "/v1/audio/speech" and status_label in ("200", "499"):
                    totals[f"speech_{status_label}"] += value
                elif route == "/v1/audio/transcriptions":
                    totals["transcriptions"] += value
            elif (
                name == "voicebox_generations_total"
                and labels.get("kind") == "stream"
                and status_label in ("cancelled", "abandoned")
            ):
                totals[f"stream_{status_label}"] += value
        return totals


class LiveKitServer:
    """livekit-server in Docker with the dev key pair (devkey/secret) and node_ip 127.0.0.1."""

    def __init__(self, args: argparse.Namespace, out: Path) -> None:
        self.args = args
        self.out = out
        self.base = args.port_base
        self.url = f"ws://127.0.0.1:{self.base}"
        self.container_id: str | None = None

    def start(self) -> float:
        b = self.base
        config = HERE / "livekit.yaml"
        if b != 7880:
            config = self.out / "livekit.yaml"
            config.write_text(
                f"port: {b}\nrtc:\n  udp_port: {b + 2}\n  tcp_port: {b + 1}\n"
                "  use_external_ip: false\n  node_ip: 127.0.0.1\n"
            )
        ports = ((b, False), (b + 1, False), (b + 2, True))
        busy = [p for p, udp in ports if port_busy(p, udp=udp)]
        if busy:
            raise SystemExit(f"ports {busy} are in use; free them or pass --port-base")
        t0 = time.monotonic()
        cmd = [
            "docker", "run", "--rm", "-d", "--name", self.args.container,
            "-p", f"{b}:{b}", "-p", f"{b + 1}:{b + 1}", "-p", f"{b + 2}:{b + 2}/udp",
            "-v", f"{config}:/etc/livekit.yaml:ro",
            self.args.image, "--dev", "--bind", "0.0.0.0", "--config", "/etc/livekit.yaml",
        ]  # fmt: skip
        started = subprocess.run(cmd, capture_output=True, text=True)
        if started.returncode != 0:
            raise SystemExit(f"docker run failed: {started.stderr.strip()}")
        # Use the returned ID, not the reusable name: teardown owns only this container.
        self.container_id = started.stdout.strip()
        while time.monotonic() - t0 < 30:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{b}/", timeout=1) as resp:
                    if resp.status == 200:
                        return time.monotonic() - t0
            except OSError:
                time.sleep(0.2)
        raise SystemExit("livekit-server did not answer within 30 s")

    def stop(self) -> None:
        if not self.container_id:
            return
        logs = subprocess.run(["docker", "logs", self.container_id], capture_output=True, text=True)
        (self.out / "livekit.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "stop", "-t", "3", self.container_id], capture_output=True)
        self.container_id = None


class Worker:
    """The agent worker process (its whole process group), logging to a file."""

    def __init__(self, cmd: list[str], env: dict[str, str], logfile: Path) -> None:
        self.logfile = logfile
        self._fh = logfile.open("w")
        self.proc = subprocess.Popen(
            cmd,
            cwd=HERE,
            env=env,
            stdout=self._fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_registered(self, timeout: float) -> float:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"the worker exited (rc={self.proc.returncode}); see {self.logfile}"
                )
            if "registered worker" in self.logfile.read_text(errors="replace"):
                return time.monotonic() - t0
            time.sleep(0.2)
        raise SystemExit(f"the worker did not register within {timeout:.0f} s; see {self.logfile}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self._signal(force=False)
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._signal(force=True)
                self.proc.wait(timeout=5)
            except ProcessLookupError:
                pass
        self._fh.close()

    def _signal(self, *, force: bool) -> None:
        """Stop the worker and its job processes (its process group); Windows has no groups."""
        if hasattr(os, "killpg"):
            os.killpg(self.proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            self.proc.kill()
        else:
            self.proc.terminate()


class Session:
    """One participant session: run it, then check what was said and heard."""

    def __init__(self, run: Run, index: int) -> None:
        self.run = run
        self.args = run.args
        self.index = index
        self.room = f"e2e-{run.run_id}-{index}"
        self.dir = run.out / f"session-{index}"
        self.failures: list[str] = []
        self.timings: dict[str, float | None] = {}
        self.evidence: dict[str, Any] = {}

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(message)

    def execute(self) -> dict[str, Any]:
        run, args = self.run, self.args
        before = run.voicebox.metrics()
        vb_offset = run.voicebox_log.stat().st_size if run.voicebox_log else 0
        cmd = [
            PY, "participant.py", "--room", self.room, "--speak", str(run.user_wav),
            "--out-dir", str(self.dir), "--dispatch-agent", run.agent_name,
            "--identity", USER_IDENTITY,
        ]  # fmt: skip
        if args.barge_in:
            cmd += ["--barge-in", str(run.barge_wav), "--barge-after", str(args.barge_after)]
        env = dict(os.environ, LIVEKIT_URL=run.livekit.url, LIVEKIT_API_KEY="devkey")
        env.update(
            LIVEKIT_API_SECRET="secret", PYTHONWARNINGS="ignore", PYTHONDONTWRITEBYTECODE="1"
        )
        started = time.monotonic()
        try:
            done = subprocess.run(
                cmd, cwd=HERE, env=env, capture_output=True, text=True, timeout=240
            )
            output = done.stdout + done.stderr
            if done.returncode:
                self.failures.append(f"participant exited with status {done.returncode}")
        except subprocess.TimeoutExpired as e:
            output = f"participant timed out\n{e.stdout or ''}{e.stderr or ''}"
            self.failures.append("participant timed out")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "participant.log").write_text(output)
        wall_s = time.monotonic() - started
        time.sleep(1.5)  # the agent's last log lines and Voicebox's last counters
        after = run.voicebox.metrics()

        report_path = self.dir / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if "phases" not in report or "error" in report:
            self.failures.append(f"participant: {report.get('error', 'no report')}")
        else:
            try:
                self._check_audio(report)
                self._check_agent(report)
                if args.barge_in:
                    self._check_barge_in(report, before, after, vb_offset)
            except Exception as e:  # a broken check fails this session, not the whole run
                self.failures.append(f"check raised {e!r}")
        if before is not None and after is not None:
            self.evidence["voicebox_delta"] = {k: after[k] - before[k] for k in after}
        summary = {
            "session": self.index,
            "room": self.room,
            "tts_transport": args.tts_transport,
            "pass": not self.failures,
            "wall_s": round(wall_s, 2),
            "timings": {k: None if v is None else round(v, 3) for k, v in self.timings.items()},
            "evidence": self.evidence,
            "failures": self.failures,
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def _transcript(self, name: str) -> str | None:
        path = self.dir / name
        if not path.exists() or path.stat().st_size < 1000:
            return None
        text = self.run.voicebox.transcribe(path)
        self.evidence[f"{name}_transcript"] = text
        return text

    def _check_text(self, what: str, expected: str, heard: str | None) -> None:
        if heard is None:
            self.failures.append(f"{what}: nothing recorded")
            return
        score = wer(expected, heard)
        self.evidence[f"{what}_wer"] = round(score, 3)
        limit = self.args.max_wer
        self.check(score <= limit, f"{what}: WER {score:.2f} > {limit}: {heard!r}")

    def _check_audio(self, report: dict[str, Any]) -> None:
        """What the participant recorded: the greeting, the reply and when it started."""
        phases = report["phases"]
        self._check_text("greeting", self.run.greeting, self._transcript("greeting.wav"))
        latency = phases["reply_start"] - phases["user_end"]
        self.timings["user_end_to_reply_s"] = latency
        limit = self.args.max_reply_latency
        self.check(0 < latency <= limit, f"reply started {latency:.2f} s after the user stopped")
        overlap = report.get("agent_audio_during_user_s", 0.0)
        self.check(overlap <= 0.1, f"agent spoke for {overlap:.2f} s while the user was talking")
        if self.args.barge_in:
            heard = self._transcript("barge-reply.wav")
            self._check_text("barge_reply", f"You said: {BARGE_LINE}", heard)
            self.timings["barge_end_to_reply_s"] = phases["barge_reply_start"] - phases["barge_end"]
        else:
            self._check_text("reply", f"You said: {USER_LINE}", self._transcript("reply.wav"))

    def _check_agent(self, report: dict[str, Any]) -> None:
        """What the agent heard, and when, from its own log."""
        user_end_wall = report["t0_wall"] + report["phases"]["user_end"]
        expected = [USER_LINE, BARGE_LINE] if self.args.barge_in else [USER_LINE]
        records = [r for r in log_records(self.run.worker.logfile) if r.get("room") == self.room]
        if self.args.agent == "example":
            turn = [r["message"] for r in records if r["message"].startswith("turn detection: ")]
            self.evidence["turn_handling"] = turn
            self.check(turn == [LOCAL_TURN_HANDLING], f"turn-taking is not local: {turn!r}")
            heard = [r for r in records if r["message"].startswith("heard: ")]
            texts = [r["message"][len("heard: ") :] for r in heard]
            self.evidence["agent_heard"] = texts
            self._check_text("agent_transcript", " ".join(expected), " ".join(texts) or None)
            reply_wall = report["t0_wall"] + report["phases"]["reply_start"]
            before_reply = [r["wall"] for r in heard if r["wall"] <= reply_wall]
            if before_reply:  # the final transcript that the reply answered
                self.timings["stt_final_s"] = before_reply[-1] - user_end_wall
            return

        events = [r["e2e"] for r in records if isinstance(r.get("e2e"), dict)]
        config = next((e for e in events if e["event"] == "config"), {})
        self.evidence["turn_detector"] = config.get("turn_detector")
        self.evidence["interruption"] = config.get("interruption")
        local = (config.get("turn_detector"), config.get("interruption")) == (
            "turn-detector-v1-mini",
            "vad",
        )
        self.check(local, f"turn-taking is not local: config {config!r}")
        users = [e for e in events if e["event"] == "item" and e["role"] == "user"]
        self.evidence["agent_user_turns"] = [u["text"] for u in users]
        if len(users) < len(expected):
            self.failures.append(f"agent logged {len(users)} user turns, expected {len(expected)}")
            return
        for n, (line, turn) in enumerate(zip(expected, users, strict=False), 1):
            self._check_text(f"agent_transcript_{n}", line, turn["text"])

        first = users[0]
        finals = [
            e
            for e in events
            if e["event"] == "transcript" and e["final"] and e["at"] <= first["at"]
        ]
        if finals:
            self.timings["stt_final_s"] = finals[-1]["at"] - user_end_wall
        self.timings["transcription_delay_s"] = first["metrics"].get("transcription_delay")
        self.timings["end_of_turn_delay_s"] = first["metrics"].get("end_of_turn_delay")
        tts = [e for e in events if e["event"] == "metrics" and e["type"] == "tts_metrics"]
        reply_tts = [e for e in tts if e["wall"] >= first["at"]]
        if reply_tts:
            self.timings["tts_ttfb_s"] = reply_tts[0]["ttfb"]
        replies = [
            e
            for e in events
            if e["event"] == "item" and e["role"] == "assistant" and e["at"] >= first["at"]
        ]
        if replies:
            self.timings["llm_ttft_s"] = replies[0]["metrics"].get("llm_node_ttft")
            self.evidence["first_reply_interrupted"] = replies[0]["interrupted"]
        self.evidence["tts_requests"] = len(tts)
        self.evidence["tts_cancelled"] = len([e for e in tts if e["cancelled"]])

    def _check_barge_in(
        self,
        report: dict[str, Any],
        before: dict[str, float] | None,
        after: dict[str, float] | None,
        vb_offset: int,
    ) -> None:
        phases = report["phases"]
        stop = phases["agent_stop"] - phases["barge_start"]
        self.timings["barge_stop_s"] = stop
        self.check(stop <= self.args.max_stop, f"agent kept talking {stop:.2f} s into the barge-in")
        resumed = report.get("agent_audio_during_barge_s", 0.0)
        self.check(resumed <= 0.1, f"agent spoke again for {resumed:.2f} s during the barge-in")
        if self.args.agent == "e2e":
            interrupted = self.evidence.get("first_reply_interrupted") is True
            self.check(interrupted, "first reply not marked interrupted")
        # A request cut before its first chunk ends as a 499; one cut mid-stream
        # shows up as a cancelled or abandoned streaming job.
        abandoned = []
        if before is not None and after is not None:
            keys = ("speech_499", "stream_cancelled", "stream_abandoned")
            abandoned.append(sum(after[k] - before[k] for k in keys))
        if self.run.voicebox_log:
            text = self.run.voicebox_log.read_bytes()[vb_offset:].decode(errors="replace")
            hits = [
                ANSI.sub("", line).strip() for line in text.splitlines() if ABANDON_LOG_LINE in line
            ]
            self.evidence["voicebox_log"] = hits
            abandoned.append(len(hits))
        if abandoned:
            self.check(max(abandoned) >= 1, "Voicebox shows no abandoned speech request")
        else:
            self.evidence["voicebox_side"] = "not checked: no admin key and no --voicebox-log"
            cancelled = self.evidence.get("tts_cancelled", 0) >= 1
            self.check(cancelled or self.args.agent == "example", "agent cancelled no TTS request")


class Run:
    """Server, worker and sessions for one invocation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        # A worker left behind by an earlier run (one killed with SIGKILL) may reconnect to
        # this run's server; under its own name it never receives this run's dispatches.
        self.agent_name = f"{AGENT_NAME}-{self.run_id}"
        prefix = f"voicebox-livekit-e2e-{self.run_id}-"
        # Child processes run from HERE, so every path passed to them must be absolute.
        self.out = Path(args.out_dir or tempfile.mkdtemp(prefix=prefix)).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.voicebox = Voicebox(args)
        self.voicebox_log = Path(args.voicebox_log) if args.voicebox_log else None
        if self.voicebox_log and not self.voicebox_log.exists():
            raise SystemExit(f"--voicebox-log {self.voicebox_log} does not exist")
        self.livekit = LiveKitServer(args, self.out)
        self.greeting = example_greeting()
        self.user_wav = self.out / "user.wav"
        self.barge_wav = self.out / "barge.wav"
        self.worker: Worker | None = None
        self.echo_server = None

    def worker_env(self) -> dict[str, str]:
        args = self.args
        env = dict(os.environ)
        env.update(
            LIVEKIT_URL=self.livekit.url,
            LIVEKIT_API_KEY="devkey",
            LIVEKIT_API_SECRET="secret",
            LIVEKIT_AGENT_NAME=self.agent_name,
            # LiveKit's console/dev defaults: its cloud turn detector and adaptive
            # interruptions, unless build_session() pins local turn-taking.
            LIVEKIT_DEV_MODE="1",
            VOICEBOX_URL=self.voicebox.url,
            VOICEBOX_CLIENT_KEY=args.client_key,
            VOICEBOX_VOICE=args.voice,
            VOICEBOX_TTS_MODEL=args.tts_model,
            VOICEBOX_TTS_TRANSPORT=args.tts_transport,
            VOICEBOX_STT_MODEL=args.stt_model,
            VOICEBOX_STT_REALTIME="1" if args.stt == "realtime" else "0",
            VOICEBOX_LANGUAGE=args.language,
            E2E_USER_IDENTITY=USER_IDENTITY,
            E2E_ECHO_TAIL=ECHO_TAIL if args.barge_in else "",
            COLUMNS="2000",  # console-formatted logs: one record per line
            PYTHONWARNINGS="ignore",
            PYTHONDONTWRITEBYTECODE="1",
        )
        return env

    def start_worker(self) -> None:
        env = self.worker_env()
        if self.args.agent == "example":
            from echo import EchoChatServer  # lazy: imports livekit.agents

            self.echo_server = EchoChatServer(tail=ECHO_TAIL if self.args.barge_in else "").start()
            env.update(
                LLM_BASE_URL=self.echo_server.base_url, LLM_MODEL="echo", LLM_API_KEY="unused"
            )
            env["LIVEKIT_LOG_LEVEL"] = "DEBUG"  # the example logs what it heard at debug level
            cmd = [PY, str(EXAMPLE), self.args.example_command]
        else:
            cmd = [PY, "e2e_agent.py", "start"]
        self.worker = Worker(cmd, env, self.out / "worker.log")
        took = self.worker.wait_registered(self.args.worker_timeout)
        log(f"worker registered in {took:.2f} s: {' '.join(Path(c).name for c in cmd[1:])}")

    def execute(self) -> int:
        args = self.args
        results: list[dict[str, Any]] = []
        log(f"artifacts in {self.out}")
        for sig in STOP_SIGNALS:
            if signal.getsignal(sig) is not signal.SIG_IGN:  # under nohup, SIGHUP stays ignored
                signal.signal(sig, _exit_on_signal)
        self.voicebox.wait_ready(args.ready_timeout)
        user_s = self.voicebox.speak(USER_LINE, args.user_voice, self.user_wav)
        barge_s = self.voicebox.speak(BARGE_LINE, args.user_voice, self.barge_wav)
        log(f"Voicebox ready; user lines in {args.user_voice}: {user_s:.2f} s and {barge_s:.2f} s")
        try:
            took = self.livekit.start()
            log(f"livekit-server {args.image} ready in {took:.2f} s ({self.livekit.url})")
            self.start_worker()
            for i in range(1, args.sessions + 1):
                result = Session(self, i).execute()
                results.append(result)
                status = "PASS" if result["pass"] else "FAIL"
                log(f"session {i}: {status} {json.dumps(result['timings'])}")
        finally:
            # A second signal must not cut the teardown short.
            held = {
                sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, *STOP_SIGNALS)
            }
            if self.worker:
                self.worker.stop()
            if self.echo_server:
                self.echo_server.stop()
            self.livekit.stop()
            for sig, handler in held.items():
                if handler is not None:
                    signal.signal(sig, handler)
            log("torn down: worker and livekit-server stopped")
        (self.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        self.print_summary(results)
        passed = sum(r["pass"] for r in results)
        ok = passed == args.sessions
        log(f"RESULT {'PASS' if ok else 'FAIL'}: {passed}/{args.sessions} sessions passed")
        return 0 if ok else 1

    def print_summary(self, results: list[dict[str, Any]]) -> None:
        barge = " barge-in" if self.args.barge_in else ""
        mode = f"stt={self.args.stt} tts={self.args.tts_transport} agent={self.args.agent}{barge}"
        print(f"\n{len(results)} session(s), {mode}; timings in seconds (min / median / max):")
        for name in sorted({k for r in results for k in r["timings"]}):
            values = [r["timings"][name] for r in results if r["timings"].get(name) is not None]
            print(f"  {name:24s} {_stats(values)}")
        for r in results:
            delta = r["evidence"].get("voicebox_delta")
            cancelled = r["evidence"].get("tts_cancelled")
            print(f"  session {r['session']}: Voicebox {delta}; agent TTS cancelled: {cancelled}")
            for failure in r["failures"]:
                print(f"  session {r['session']} FAIL: {failure}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    env = os.environ.get
    ap.add_argument("--stt", choices=["realtime", "batch"], default="realtime")
    ap.add_argument("--sessions", type=int, default=1, help="sessions on one server and worker")
    ap.add_argument("--barge-in", action="store_true", help="talk over the agent's first reply")
    ap.add_argument(
        "--agent",
        choices=["e2e", "example"],
        default="e2e",
        help="e2e_agent.py with EchoLLM, or ../agent.py with an echo chat server",
    )
    ap.add_argument(
        "--example-command",
        choices=["dev", "start"],
        default="dev",
        help="how --agent example runs ../agent.py",
    )
    ap.add_argument("--voicebox-url", default=env("VOICEBOX_URL", "http://127.0.0.1:17493"))
    ap.add_argument("--client-key", default=env("VOICEBOX_CLIENT_KEY"), help="VOICEBOX_CLIENT_KEY")
    ap.add_argument(
        "--admin-key", default=env("VOICEBOX_ADMIN_KEY"), help="for /metrics (VOICEBOX_ADMIN_KEY)"
    )
    ap.add_argument("--voicebox-log", help="Voicebox's log file, searched for abandoned streams")
    ap.add_argument("--voice", default=env("VOICEBOX_VOICE", "af_heart"), help="agent's voice")
    ap.add_argument("--user-voice", default="am_adam", help="voice of the user's lines")
    ap.add_argument(
        "--tts-model", default=env("VOICEBOX_TTS_MODEL", "tts-1"), help="as ../agent.py's default"
    )
    ap.add_argument(
        "--tts-transport",
        choices=["http", "websocket"],
        default=env("VOICEBOX_TTS_TRANSPORT", "http"),
    )
    ap.add_argument("--stt-model", default=env("VOICEBOX_STT_MODEL", "whisper-1"))
    ap.add_argument("--language", default=env("VOICEBOX_LANGUAGE", "en"))
    ap.add_argument("--max-wer", type=float, default=0.25)
    ap.add_argument("--max-reply-latency", type=float, default=5.0, help="user end to reply, s")
    ap.add_argument("--max-stop", type=float, default=1.5, help="barge-in to agent silence, s")
    ap.add_argument("--barge-after", type=float, default=1.5, help="reply start to barge-in, s")
    ap.add_argument("--port-base", type=int, default=7880, help="+1 is ICE/TCP, +2 ICE/UDP")
    ap.add_argument("--image", default="livekit/livekit-server:v1.13.10")
    ap.add_argument("--container", default="lk-e2e")
    ap.add_argument("--ready-timeout", type=float, default=120.0)
    ap.add_argument("--worker-timeout", type=float, default=120.0)
    ap.add_argument("--out-dir", help="artifacts; default: a new temporary directory")
    args = ap.parse_args(argv)
    if not args.client_key:
        ap.error("set VOICEBOX_CLIENT_KEY or pass --client-key")
    if args.sessions < 1:
        ap.error("--sessions must be at least 1")
    return args


def main() -> int:
    return Run(parse_args()).execute()


if __name__ == "__main__":
    sys.exit(main())
