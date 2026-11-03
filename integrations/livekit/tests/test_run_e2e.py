"""run_e2e.py's own machinery, without Docker, LiveKit or Voicebox."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_e2e

E2E_DIR = Path(run_e2e.__file__).resolve().parent
REPORT = {"t0_wall": 0.0, "phases": {"user_end": 1.0, "reply_start": 2.0}}
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and process groups")


def make_run(tmp_path: Path, *argv: str) -> run_e2e.Run:
    args = ["--client-key", "vbx_test", "--admin-key", "", "--out-dir", str(tmp_path), *argv]
    return run_e2e.Run(run_e2e.parse_args(args))


def write_worker_log(run: run_e2e.Run, room: str, records: list[dict]) -> None:
    """Records as the worker logs them in ``start`` mode: JSON lines tagged with the room."""
    run.worker = SimpleNamespace(logfile=run.out / "worker.log")
    stamp = "2026-10-17T12:00:00+00:00"
    lines = [json.dumps({"timestamp": stamp, "room": room, **record}) for record in records]
    run.worker.logfile.write_text("\n".join(lines) + "\n")


def test_the_worker_runs_with_livekits_dev_defaults_under_this_runs_name(tmp_path):
    run = make_run(tmp_path)
    env = run.worker_env()
    assert env["LIVEKIT_DEV_MODE"] == "1"  # so only build_session() keeps turn-taking local
    assert env["LIVEKIT_AGENT_NAME"] == run.agent_name == f"e2e-agent-{run.run_id}"


def test_the_caller_dispatches_this_runs_worker(tmp_path, monkeypatch):
    """A worker left over from a killed run never takes this run's calls."""
    run = make_run(tmp_path)
    commands = []

    def participant(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(subprocess, "run", participant)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    run_e2e.Session(run, 1).execute()
    [cmd] = commands
    assert cmd[cmd.index("--dispatch-agent") + 1] == run.agent_name


def test_relative_output_paths_work_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = make_run(Path("artifacts"))
    assert run.out == tmp_path / "artifacts"
    assert run.user_wav.is_absolute()


@pytest.mark.parametrize("timeout", [False, True])
def test_a_failed_participant_cannot_pass_with_an_existing_report(tmp_path, monkeypatch, timeout):
    run = make_run(tmp_path)
    session = run_e2e.Session(run, 1)
    session.dir.mkdir()
    (session.dir / "report.json").write_text(json.dumps(REPORT))

    def participant(cmd, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(cmd, 240)
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(subprocess, "run", participant)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(session, "_check_audio", lambda report: None)
    monkeypatch.setattr(session, "_check_agent", lambda report: None)
    result = session.execute()
    assert not result["pass"]
    assert result["failures"] == [
        "participant timed out" if timeout else "participant exited with status 1"
    ]


def test_a_failed_container_start_leaves_existing_containers_alone(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    commands = []

    def docker(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "container name is already in use")

    monkeypatch.setattr(subprocess, "run", docker)
    monkeypatch.setattr(run_e2e, "port_busy", lambda port, udp=False: False)
    try:
        with pytest.raises(SystemExit, match="container name is already in use"):
            run.livekit.start()
    finally:
        run.livekit.stop()
    assert len(commands) == 1
    assert commands[0][:2] == ["docker", "run"]


def test_teardown_stops_only_the_container_started_by_this_run(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    commands = []

    def docker(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "own-container-id\n", "")

    class Ready:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(subprocess, "run", docker)
    monkeypatch.setattr(run_e2e, "port_busy", lambda port, udp=False: False)
    monkeypatch.setattr(run_e2e.urllib.request, "urlopen", lambda *args, **kwargs: Ready())
    run.livekit.start()
    run.livekit.stop()
    run.livekit.stop()  # repeated teardown is harmless
    assert commands[1:] == [
        ["docker", "logs", "own-container-id"],
        ["docker", "stop", "-t", "3", "own-container-id"],
    ]


@pytest.mark.parametrize(
    ("config", "local"),
    [
        ({"turn_detector": "turn-detector-v1-mini", "interruption": "vad"}, True),
        ({"turn_detector": "turn-detector-v1", "interruption": "vad"}, False),  # LiveKit Cloud
        ({"turn_detector": "turn-detector-v1-mini", "interruption": "NOT_GIVEN"}, False),
        ({}, False),
    ],
)
def test_the_e2e_worker_must_report_local_turn_taking(tmp_path, config, local):
    run = make_run(tmp_path)
    session = run_e2e.Session(run, 1)
    event = {"event": "config", "wall": 0.0, **config}
    write_worker_log(run, session.room, [{"message": "E2E config", "e2e": event}])
    session._check_agent(REPORT)
    flagged = [f for f in session.failures if f.startswith("turn-taking is not local")]
    assert bool(flagged) is not local


@pytest.mark.parametrize(
    ("line", "local"),
    [
        (run_e2e.LOCAL_TURN_HANDLING, True),
        ("turn detection: turn-detector-v1, interruptions: NOT_GIVEN", False),
        (None, False),
    ],
)
def test_the_example_must_log_local_turn_taking(tmp_path, line, local):
    run = make_run(tmp_path, "--agent", "example")
    session = run_e2e.Session(run, 1)
    records = [{"message": f"heard: {run_e2e.USER_LINE}"}]
    write_worker_log(run, session.room, records if line is None else [{"message": line}, *records])
    session._check_agent(REPORT)
    flagged = [f for f in session.failures if f.startswith("turn-taking is not local")]
    assert bool(flagged) is not local


def test_ports_are_probed_without_lsof(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing to run on the PATH
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert run_e2e.port_busy(port)
    assert not run_e2e.port_busy(port)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.bind(("0.0.0.0", 0))
        port = udp.getsockname()[1]
        assert run_e2e.port_busy(port, udp=True)
    assert not run_e2e.port_busy(port, udp=True)


def test_a_port_left_in_time_wait_is_free():
    """What a stopped container leaves behind: closed connections in TIME_WAIT, no listener."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        accepted, _ = listener.accept()
        accepted.close()  # the server's side closes first, so it is the one in TIME_WAIT
        client.close()
    assert not run_e2e.port_busy(port)


def test_the_worker_stops_without_process_groups(tmp_path, monkeypatch):
    """As on Windows, which has no os.killpg."""
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    worker = run_e2e.Worker(sleeper, dict(os.environ), tmp_path / "worker.log")
    try:
        monkeypatch.delattr(os, "killpg", raising=False)
        worker.stop()
        assert worker.proc.returncode is not None
    finally:
        if worker.proc.poll() is None:
            worker.proc.kill()
            worker.proc.wait()


DRIVER = """
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import run_e2e

marks = Path(sys.argv[2])


class Voicebox:
    url = "http://127.0.0.1:9"

    def __init__(self, args):
        pass

    def wait_ready(self, timeout):
        pass

    def speak(self, text, voice, path):
        return 1.0

    def metrics(self):
        return None


class LiveKitServer:
    url = "ws://127.0.0.1:9"

    def __init__(self, args, out):
        pass

    def start(self):
        return 0.0

    def stop(self):
        (marks / "livekit-stopped").touch()


def start_worker(run):
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    run.worker = run_e2e.Worker(sleeper, run.worker_env(), run.out / "worker.log")
    (marks / "worker-pid").write_text(str(run.worker.proc.pid))


def session_in_progress(session):
    (marks / "session-started").touch()
    time.sleep(120)


run_e2e.Voicebox = Voicebox
run_e2e.LiveKitServer = LiveKitServer
run_e2e.Run.start_worker = start_worker
run_e2e.Session.execute = session_in_progress
args = run_e2e.parse_args(["--client-key", "x", "--admin-key", "", "--out-dir", str(marks / "run")])
sys.exit(run_e2e.Run(args).execute())
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(condition, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


@POSIX
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_run_ended_by_a_signal_stops_its_worker_and_server(tmp_path, name):
    """kill, timeout and a closed terminal end a run as Ctrl-C does: through its teardown."""
    signum = getattr(signal, name)
    if signal.getsignal(signum) is signal.SIG_IGN:
        pytest.skip(f"this test runner ignores {name}, and so would the run (nohup)")
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    marks = tmp_path / "marks"
    marks.mkdir()
    proc = subprocess.Popen([sys.executable, str(driver), str(E2E_DIR), str(marks)])
    worker_pid = None
    try:
        wait_for(lambda: (marks / "session-started").exists(), 30)
        worker_pid = int((marks / "worker-pid").read_text())
        proc.send_signal(signum)
        assert proc.wait(timeout=30) == 128 + signum
        assert (marks / "livekit-stopped").exists()
        assert not alive(worker_pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if worker_pid is not None and alive(worker_pid):
            os.killpg(worker_pid, signal.SIGKILL)
