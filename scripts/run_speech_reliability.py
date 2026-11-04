"""Run model-backed voice-agent checks against an isolated, disposable Voicebox container.

Uses separately installed LiveKit/Pipecat interpreters; never changes a host server.
The container is restarted between passes to exercise recovery with persisted keys.
"""

import argparse
import json
import os
import secrets
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(url, *, key=None, data=None):
    """Make a bounded local request without putting credentials in command arguments."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode() if data else None)
    with urllib.request.urlopen(req, timeout=10) as response:
        body = response.read()
    return json.loads(body) if body else None


def ready(url, timeout):
    """Wait for explicit model preloads after boot or an owned-container restart."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request(url + "/health/ready")
            return
        except (OSError, urllib.error.HTTPError):
            time.sleep(1)
    raise RuntimeError("Voicebox did not become ready within the model preload timeout")


def read_metrics(url, key):
    req = urllib.request.Request(url + "/metrics", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and not parts[0].startswith("#"):
            values[parts[0]] = float(parts[1])
    return values


def speech_connections(url, key):
    values = read_metrics(url, key)
    return {
        "accepted": values.get('voicebox_websocket_accepted_total{kind="speech"}', 0),
        "idle_closed": values.get('voicebox_websocket_closed_total{kind="speech",reason="idle_timeout"}', 0),
    }


def reconnect_evidence(folder, before, after):
    """A successful same-call turn after server idle closure proves a fresh connection."""
    turns = len(json.loads((folder / "summary.json").read_text())["turns"])
    counts = {name: after[name] - before[name] for name in before}
    if turns > 1 and (counts["idle_closed"] < turns - 1 or counts["accepted"] < turns):
        raise RuntimeError(f"Pipecat did not reconnect across {turns} turns: {counts}")
    return {**counts, "turns": turns, "checked": turns > 1}


def wait_idle(url, key, timeout=30):
    """Require queue/session gauges to return to zero after each disconnected call."""
    wanted = {
        "voicebox_queue_pending_jobs",
        "voicebox_queue_running_jobs",
        "voicebox_realtime_sessions",
        "voicebox_websocket_sessions",
        "voicebox_speech_pending_flushes",
    }
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        values = {name: value for name, value in read_metrics(url, key).items() if name.split("{", 1)[0] in wanted}
        found = {name.split("{", 1)[0] for name in values}
        if not wanted <= found:
            raise RuntimeError("Required Voicebox queue/session metrics are missing")
        if all(value == 0 for value in values.values()):
            return values
        time.sleep(0.5)
    raise RuntimeError(f"Call leaked active work after disconnect: {values}")


def execute(command, *, env, log, timeout):
    """Terminate a timed-out/interrupted driver gracefully so it cleans up its own workers."""
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)


def stop_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def run(args):
    """Return nonzero for any failed conversation; remove only this run's container."""
    out = Path(args.out_dir or tempfile.mkdtemp(prefix="voicebox-reliability-")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = "voicebox-reliability-" + secrets.token_hex(6)
    admin = secrets.token_urlsafe(32)
    environment = out / ".container-env"
    environment.touch(mode=0o600, exist_ok=False)
    environment.write_text(
        f"VOICEBOX_API_KEY={admin}\nVOICEBOX_PRELOAD_MODELS=kokoro,whisper-base\n"
        "VOICEBOX_REALTIME_IDLE_S=15\nVOICEBOX_SPEECH_IDLE_S=10\n"
        "VOICEBOX_LOG_FORMAT=json\nVOICEBOX_REPLICA_THREADS=2\n"
    )
    container_id = None
    results = []
    try:
        container_id = subprocess.check_output(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-p",
                "127.0.0.1::17493",
                "--env-file",
                str(environment),
                args.image,
            ],
            text=True,
        ).strip()
        address = subprocess.check_output(["docker", "port", container_id, "17493/tcp"], text=True).strip()
        url = "http://" + address
        ready(url, args.ready_timeout)
        key = request(
            url + "/auth/keys",
            key=admin,
            data={
                "id": "reliability",
                "role": "client",
                "limits": {
                    "requests": 10000,
                    "inference": 10000,
                    "tts_chars": 1000000,
                    "max_pending_jobs": 32,
                    "max_realtime_sessions": 16,
                },
            },
        )["key"]
        env = dict(os.environ, VOICEBOX_URL=url, VOICEBOX_CLIENT_KEY=key, VOICEBOX_ADMIN_KEY=admin)
        # External account credentials cannot accidentally turn the CI smoke into paid traffic.
        env.pop("VOICEBOX_TTS_TIMEOUT", None)
        for number in range(1, args.rounds + 1):
            if number > 1:
                subprocess.run(["docker", "restart", "-t", "45", container_id], check=True, stdout=subprocess.DEVNULL)
                # Docker can assign a new ephemeral host port on restart (notably Desktop).
                address = subprocess.check_output(["docker", "port", container_id, "17493/tcp"], text=True).strip()
                url = "http://" + address
                env["VOICEBOX_URL"] = url
                ready(url, args.ready_timeout)
            for integration, python in [("livekit", args.livekit_python), ("pipecat", args.pipecat_python)]:
                folder = out / f"{integration}-{number}"
                session_timeout = max(900, int(args.duration) + args.turns * 240 + 180)
                if integration == "livekit":
                    command = [
                        python,
                        "integrations/livekit/e2e/run_e2e.py",
                        "--tts-transport",
                        args.tts,
                        "--container",
                        name + "-livekit",
                        "--port-base",
                        str(args.livekit_port),
                        "--session-timeout",
                        str(session_timeout),
                    ]
                else:
                    command = [python, "integrations/pipecat/e2e.py", "--tts", args.tts, "--timeout", "120"]
                command += [
                    "--llm",
                    "echo",
                    "--turns",
                    str(args.turns),
                    "--duration",
                    str(args.duration),
                    "--idle-between",
                    "18",
                    "--barge-in",
                    "--max-reply-latency",
                    str(args.max_reply_latency),
                    "--out-dir",
                    str(folder),
                ]
                print(f"{integration} pass {number}: {folder}", flush=True)
                before = (
                    speech_connections(url, admin) if integration == "pipecat" and args.tts == "websocket" else None
                )
                with (out / f"{integration}-{number}.log").open("w") as log:
                    # A soak may begin its final room just before the minimum duration.
                    exit_code = execute(command, env=env, log=log, timeout=args.duration + session_timeout + 60)
                result = {"integration": integration, "round": number, "exit_code": exit_code}
                results.append(result)
                (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                if exit_code:
                    raise RuntimeError(f"{integration} failed; inspect {folder} and its log")
                try:
                    result["idle_gauges"] = wait_idle(url, admin)
                    if before is not None:
                        result["idle_reconnect"] = reconnect_evidence(folder, before, speech_connections(url, admin))
                except Exception as exc:
                    result["error"] = str(exc)
                    raise
                finally:
                    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        return 0
    finally:
        environment.unlink(missing_ok=True)
        if container_id:
            with (out / "voicebox.log").open("w") as log:
                subprocess.run(["docker", "logs", container_id], stdout=log, stderr=subprocess.STDOUT, check=False)
            subprocess.run(["docker", "rm", "-f", container_id], stdout=subprocess.DEVNULL, check=False)
        print(f"Artifacts: {out}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="local CPU image built from this checkout")
    parser.add_argument("--livekit-python", required=True)
    parser.add_argument("--pipecat-python", required=True)
    parser.add_argument("--tts", choices=["http", "websocket"], default="websocket")
    parser.add_argument("--rounds", type=int, default=2, help="restart the owned server between rounds")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--duration", type=float, default=180, help="minimum seconds per integration/pass")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--max-reply-latency", type=float, default=30, help="CPU CI allowance, not an SLO")
    parser.add_argument("--livekit-port", type=int, default=7980)
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 10 or not 1 <= args.turns <= 100 or not 0 <= args.duration <= 3600:
        parser.error("rounds 1..10, turns 1..100, duration 0..3600 required")
    if not 0 < args.ready_timeout <= 3600 or not 0 < args.max_reply_latency <= 120:
        parser.error("ready-timeout must be 0..3600 and max-reply-latency 0..120 (exclusive of zero)")
    if not 1024 <= args.livekit_port <= 65532:
        parser.error("livekit-port must be 1024..65532, leaving room for adjacent ports")
    return args


if __name__ == "__main__":
    for signal_name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), stop_on_signal)
    raise SystemExit(run(parse_args()))
