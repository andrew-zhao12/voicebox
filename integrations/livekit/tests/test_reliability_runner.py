"""Isolation and cleanup checks for the model-backed reliability runner."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_speech_reliability.py"
spec = importlib.util.spec_from_file_location("speech_reliability", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def arguments(tmp_path):
    return runner.parse_args(
        [
            "--image",
            "voicebox:test",
            "--livekit-python",
            sys.executable,
            "--pipecat-python",
            sys.executable,
            "--out-dir",
            str(tmp_path),
            "--rounds",
            "2",
            "--turns",
            "1",
            "--duration",
            "0",
        ]
    )


def test_docker_launch_failure_never_removes_existing_container(tmp_path, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "check_output", fail)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(subprocess.CalledProcessError):
        runner.run(arguments(tmp_path))
    assert not calls
    assert not (tmp_path / ".container-env").exists()


def test_restart_checks_both_integrations_and_removes_only_owned_container(tmp_path, monkeypatch):
    commands, executions = [], []
    readiness, idle_checks, speech_checks = [], [], []
    addresses = iter(["127.0.0.1:45678\n", "127.0.0.1:45679\n"])

    def output(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            environment = Path(command[command.index("--env-file") + 1])
            assert "VOICEBOX_SPEECH_IDLE_S=10\n" in environment.read_text()
            assert environment.stat().st_mode & 0o777 == 0o600
        return "own-id\n" if command[1] == "run" else next(addresses)

    def execute(command, **kwargs):
        executions.append((command, dict(kwargs["env"])))
        return 0

    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(runner, "ready", lambda url, *a: readiness.append(url))
    monkeypatch.setattr(runner, "request", lambda *a, **kw: {"key": "private-client"})
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(
        runner,
        "wait_idle",
        lambda url, *a: idle_checks.append(url) or {"voicebox_queue_pending_jobs": 0},
    )
    monkeypatch.setattr(
        runner,
        "speech_connections",
        lambda url, *a: speech_checks.append(url) or {"accepted": 0, "idle_closed": 0},
    )
    monkeypatch.setattr(runner, "reconnect_evidence", lambda *a: {"checked": True})
    args = arguments(tmp_path)
    args.duration = 1800
    args.turns = 60
    assert runner.run(args) == 0
    assert [command[1] for command, _ in executions] == [
        "integrations/livekit/e2e/run_e2e.py",
        "integrations/pipecat/e2e.py",
        "integrations/livekit/e2e/run_e2e.py",
        "integrations/pipecat/e2e.py",
    ]
    assert all("private-client" not in " ".join(command) for command, _ in executions)
    assert all(env["VOICEBOX_CLIENT_KEY"] == "private-client" for _, env in executions)
    assert [env["VOICEBOX_URL"] for _, env in executions] == [
        "http://127.0.0.1:45678",
        "http://127.0.0.1:45678",
        "http://127.0.0.1:45679",
        "http://127.0.0.1:45679",
    ]
    assert readiness == ["http://127.0.0.1:45678", "http://127.0.0.1:45679"]
    assert idle_checks == speech_checks == [url for url in readiness for _ in range(2)]
    for command, _ in executions:
        if "--session-timeout" in command:
            assert int(command[command.index("--session-timeout") + 1]) > 1800 + 60 * 240
    assert ["docker", "restart", "-t", "45", "own-id"] in commands
    assert commands[-1] == ["docker", "rm", "-f", "own-id"]
    assert not (tmp_path / ".container-env").exists()


@pytest.mark.parametrize(
    ("accepted", "idle_closed", "passes"), [(7, 5, True), (6, 5, False), (7, 4, False)]
)
def test_reconnect_requires_idle_closes_and_fresh_connections(
    tmp_path, accepted, idle_closed, passes
):
    (tmp_path / "summary.json").write_text('{"turns": [{}, {}, {}]}')
    before = {"accepted": 4, "idle_closed": 3}
    after = {"accepted": accepted, "idle_closed": idle_closed}
    if passes:
        assert runner.reconnect_evidence(tmp_path, before, after) == {
            "accepted": 3,
            "idle_closed": 2,
            "turns": 3,
            "checked": True,
        }
    else:
        with pytest.raises(RuntimeError, match="did not reconnect"):
            runner.reconnect_evidence(tmp_path, before, after)


@pytest.mark.parametrize(
    ("option", "value"),
    [("--ready-timeout", "nan"), ("--max-reply-latency", "inf"), ("--livekit-port", "65535")],
)
def test_invalid_limits_fail_before_starting_container(option, value):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--image",
                "test",
                "--livekit-python",
                "python",
                "--pipecat-python",
                "python",
                option,
                value,
            ]
        )


def test_wait_idle_checks_every_labelled_session_and_queue_gauge(monkeypatch):
    baseline = (
        b"voicebox_queue_pending_jobs 0\nvoicebox_realtime_sessions 0\n"
        b'voicebox_queue_running_jobs{lane="local"} 0\n'
    )
    replies = iter(
        [
            baseline + b'voicebox_websocket_sessions{kind="speech"} 1\n'
            b"voicebox_speech_pending_flushes 1\n",
            baseline + b'voicebox_websocket_sessions{kind="speech"} 0\n'
            b"voicebox_speech_pending_flushes 0\n",
        ]
    )
    monkeypatch.setattr(
        runner.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(next(replies))
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    values = runner.wait_idle("http://localhost:1", "test-key")
    assert values['voicebox_websocket_sessions{kind="speech"}'] == 0
    assert values["voicebox_speech_pending_flushes"] == 0


def test_wait_idle_rejects_leaked_work_and_missing_metrics(monkeypatch):
    elapsed = [0.0]

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(runner.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(runner.time, "sleep", sleep)
    body = (
        b"voicebox_queue_pending_jobs 1\nvoicebox_realtime_sessions 0\n"
        b'voicebox_queue_running_jobs{lane="local"} 0\n'
        b'voicebox_websocket_sessions{kind="speech"} 0\n'
        b"voicebox_speech_pending_flushes 0\n"
    )
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(body))
    with pytest.raises(RuntimeError, match="leaked active work"):
        runner.wait_idle("http://localhost:1", "test-key", timeout=1)
    body = b"# Empty registry\n"
    with pytest.raises(RuntimeError, match="metrics are missing"):
        runner.wait_idle("http://localhost:1", "test-key", timeout=1)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX termination")
def test_driver_timeout_runs_graceful_cleanup(tmp_path):
    mark = tmp_path / "stopped"
    source = (
        "import signal,time,pathlib,sys\n"
        "def stop(*a):\n"
        " pathlib.Path(sys.argv[1]).write_text('stopped')\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "time.sleep(30)\n"
    )
    with (tmp_path / "driver.log").open("w") as log, pytest.raises(subprocess.TimeoutExpired):
        runner.execute(
            [sys.executable, "-c", source, str(mark)], env=dict(os.environ), log=log, timeout=0.5
        )
    assert mark.read_text() == "stopped"
