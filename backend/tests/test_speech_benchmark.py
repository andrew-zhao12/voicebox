"""Exercise the HTTP speech benchmark against a local model-free server."""

import asyncio
import csv
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from scripts import speech_benchmark as benchmark


def write_wav(path: Path, *, samples: int = 1600, width: int = 2) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(width)
        audio.setframerate(16000)
        audio.writeframes(b"\0" * samples * width)


def make_manifest(tmp_path: Path, *, real: bool = True, synthetic: bool = False) -> Path:
    audio = []
    for source, enabled in (("real", real), ("synthetic", synthetic)):
        if enabled:
            write_wav(tmp_path / f"{source}.wav")
            audio.append({"id": source, "path": f"{source}.wav", "source": source, "text": "The quick fox."})
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "stub corpus",
                "audio": audio,
                "tts": [{"id": "speech", "model": "stub", "voice": "test", "text": "An original test sentence."}],
            }
        )
    )
    return path


@pytest.fixture
def stub_server():
    state = {"active": 0, "peak": 0, "cancelled": 0, "auth": [], "models": 0, "tts": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send_body(self, body, *, status=200, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):
            state["auth"].append((self.path, self.headers.get("Authorization")))
            if self.path == "/models/status":
                state["models"] += 1
                self.send_body(b'{"models": [{"model_name": "stub", "is_loaded": true}]}')
            elif self.path == "/metrics":
                body = (
                    'voicebox_gpu_memory_bytes{kind="allocated"} 123456\n'
                    'voicebox_gpu_memory_bytes{kind="reserved"} 234567\n'
                    f'voicebox_generations_total{{kind="stream",engine="stub",status="cancelled"}} {state["cancelled"]}\n'
                )
                self.send_body(body.encode(), content_type="text/plain")
            else:
                self.send_body(b"{}", status=404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["auth"].append((self.path, self.headers.get("Authorization")))
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.015)
                if self.path == "/v1/audio/transcriptions":
                    if b"reject-model" in body:
                        self.send_body(b'{"error":"must never be stored"}', status=429)
                    else:
                        self.send_body(b'{"text": "The slow fox."}')
                elif self.path == "/v1/audio/speech":
                    state["tts"] += 1
                    request = json.loads(body)
                    model = request["model"]
                    if model == "slow":
                        time.sleep(0.1)
                    if model == "empty":
                        self.send_body(b"", content_type="application/octet-stream")
                        return
                    if model == "failure":
                        self.send_body(b'{"error":"secret-server-detail"}', status=503)
                        return
                    size = 4800 if model != "slow" else 48000 * 10
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("X-Voicebox-Sample-Rate", "24000")
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    for _ in range(size // 480):
                        self.wfile.write(b"\0" * 480)
                        self.wfile.flush()
                        if model == "broken":
                            self.close_connection = True
                            break
                        time.sleep(0.002 if model != "slow" else 0.02)
                else:
                    self.send_body(b"{}", status=404)
            except (BrokenPipeError, ConnectionResetError):
                with lock:
                    state["cancelled"] += 1
                self.close_connection = True
            finally:
                with lock:
                    state["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_normalization_and_weighted_accuracy():
    result = benchmark.accuracy("The quick fox.", "THE slow fox!")
    assert result["word_errors"] == 1
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["reference_words"] == 3
    assert result["char_errors"] == 5
    assert benchmark.normalized("\uff26\uff4f\uff4f—BAR") == "foo bar"
    assert benchmark.percentile([1, 2], 0.95) == 2
    assert benchmark.percentile([], 0.95) is None


def test_manifest_labels_and_hashes_are_reproducible(tmp_path):
    path = make_manifest(tmp_path, synthetic=True)
    manifest = benchmark.load_manifest(path)
    assert {clip["source"] for clip in manifest["audio"]} == {"real", "synthetic"}
    assert manifest["audio"][0]["audio_s"] == pytest.approx(0.1)
    assert manifest["tts"][0]["source"] == "synthetic"
    assert benchmark.load_manifest(path) == manifest
    assert len(manifest["sha256"]) == 64


@pytest.mark.parametrize("problem", ["missing_source", "duplicate", "truncated", "wrong_width", "empty", "oversize"])
def test_invalid_corpus_is_rejected_before_requests(tmp_path, problem):
    path = make_manifest(tmp_path)
    raw = json.loads(path.read_text())
    if problem == "missing_source":
        del raw["audio"][0]["source"]
    elif problem == "duplicate":
        raw["audio"].append(raw["audio"][0])
    elif problem == "truncated":
        clip = tmp_path / "real.wav"
        clip.write_bytes(clip.read_bytes()[:-10])
    elif problem == "wrong_width":
        write_wav(tmp_path / "real.wav", width=1)
    elif problem == "empty":
        write_wav(tmp_path / "real.wav", samples=0)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"source|duplicate|truncated|PCM16|size limit"):
        benchmark.load_manifest(path, max_audio_bytes=1 if problem == "oversize" else 1024**2)


def test_local_http_sweeps_write_honest_model_source_and_gpu_reports(tmp_path, stub_server, capsys):
    url, state = stub_server
    path = make_manifest(tmp_path, synthetic=True)
    manifest = benchmark.load_manifest(path)
    args = benchmark.parse_args(
        [
            str(path),
            "--output",
            str(tmp_path / "output"),
            "--url",
            url,
            "--stt-models",
            "first,second",
            "--concurrency",
            "1,2",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--sample-interval",
            "0.01",
        ]
    )
    report = asyncio.run(benchmark.run(args, manifest, key="private-client", admin_key="private-admin"))
    assert report["ok"]
    assert len(report["summaries"]) == 10  # 2 STT models * 2 sources + 1 TTS group, at 2 concurrencies
    assert len(report["requests"]) == 20
    assert all(row["source"] in {"real", "synthetic"} for row in report["summaries"])
    assert state["peak"] == 2
    for summary in report["summaries"]:
        assert summary["requests"] == summary["successes"] == 2
        assert summary["max_in_flight"] == summary["concurrency"]
        assert summary["errors"] == 0
        if summary["kind"] == "stt":
            assert summary["wer"] == pytest.approx(1 / 3)
        else:
            assert summary["audio_s"] == pytest.approx(0.2)
            assert 0 < summary["first_audio_s_p95"] <= summary["latency_s_p95"]
    assert report["observations"]["gpu_allocator_peak_bytes"] == {"allocated": 123456, "reserved": 234567}
    assert state["models"] > 0
    assert all(
        auth == "Bearer private-admin" for route, auth in state["auth"] if route in {"/metrics", "/models/status"}
    )
    assert all(auth == "Bearer private-client" for route, auth in state["auth"] if route.startswith("/v1"))
    serialized = json.dumps(report, allow_nan=False)
    assert "private-client" not in serialized + capsys.readouterr().out
    assert "private-admin" not in serialized
    csv_path = tmp_path / "summaries.csv"
    benchmark.write_csv(csv_path, report["summaries"])
    with csv_path.open() as handle:
        assert len(list(csv.DictReader(handle))) == 10


@pytest.mark.parametrize("model", ["failure", "broken", "empty"])
def test_http_errors_broken_transfers_and_empty_pcm_cannot_pass(stub_server, model):
    url, _state = stub_server
    case = {"id": "test", "text": "Test.", "voice": "test", "source": "synthetic"}

    async def run():
        async with httpx.AsyncClient(base_url=url) as client:
            row = benchmark.new_row("tts", model, case, 1, 0)
            return await benchmark.request_one(client, case, row, 2)

    row = asyncio.run(run())
    assert not row["ok"]
    assert row["error"]
    assert "secret-server-detail" not in json.dumps(row)
    summary = benchmark.summarize([row], 1, 1)
    assert summary["successes"] == 0
    assert summary["first_audio_s_p95"] is None
    assert summary["audio_per_wall_x"] == 0


def test_rate_limits_are_errors_and_missing_metrics_do_not_imply_zero_vram(tmp_path, stub_server):
    url, _state = stub_server
    path = make_manifest(tmp_path)
    manifest = benchmark.load_manifest(path)
    manifest["tts"] = []
    args = benchmark.parse_args(
        [
            str(path),
            "--output",
            str(tmp_path / "output"),
            "--url",
            url,
            "--stt-models",
            "reject-model",
            "--concurrency",
            "1",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ]
    )
    report = asyncio.run(benchmark.run(args, manifest, key="private", admin_key=None))
    assert not report["ok"]
    assert report["summaries"][0]["rate_limited"] == 1
    assert report["summaries"][0]["errors"] == 1
    assert report["observations"]["gpu_allocator_peak_bytes"] is None
    assert "unavailable" in report["observations"]["snapshots"][0]


def test_a_changed_audio_fixture_fails_instead_of_mixing_corpora(tmp_path, stub_server):
    path = make_manifest(tmp_path)
    case = benchmark.load_manifest(path)["audio"][0]
    write_wav(tmp_path / "real.wav", samples=3200)
    url, _state = stub_server

    async def run():
        async with httpx.AsyncClient(base_url=url) as client:
            row = benchmark.new_row("stt", "stub", case, 1, 0)
            return await benchmark.request_one(client, case, row, 1)

    row = asyncio.run(run())
    assert not row["ok"]
    assert row["status"] == 0


@pytest.mark.parametrize("after", [0.03, 0.15])
def test_stream_cancellation_separates_client_close_and_server_observation(stub_server, after):
    url, state = stub_server
    case = {"id": "cancel", "model": "slow", "voice": "test", "text": "Long speech.", "source": "synthetic"}

    async def run():
        async with httpx.AsyncClient(base_url=url) as client:
            return await benchmark.cancellation_probe(
                client, benchmark.Observer(client), case, after=after, observe=2, timeout=3
            )

    result = asyncio.run(run())
    assert result["cancelled"]
    assert result["request"]["error"] == "client_cancelled"
    assert result["client_close_s"] >= 0
    assert result["server_cancel_observed_s_upper_bound"] is not None
    assert result["server_counter_delta"]["stream_cancelled"] >= 1
    assert state["cancelled"] >= 1


def test_completed_requests_are_not_claimed_as_cancellations(stub_server):
    url, _state = stub_server
    case = {"id": "short", "model": "stub", "voice": "test", "text": "Short.", "source": "synthetic"}

    async def run():
        async with httpx.AsyncClient(base_url=url) as client:
            return await benchmark.cancellation_probe(
                client, benchmark.Observer(None), case, after=0.2, observe=1, timeout=2
            )

    result = asyncio.run(run())
    assert not result["cancelled"]
    assert result["request"]["ok"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--url", "https://user:secret@example.com"],
        ["--url", "https://example.com/?token=secret"],
        ["--concurrency", "0"],
        ["--concurrency", "129"],
        ["--timeout", "nan"],
        ["--cancel-after", "inf"],
        ["--repeats", "0"],
        ["--warmup", "-1"],
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["manifest.json", "--output", "out", *arguments])


def test_cli_writes_json_csv_and_refuses_to_overwrite_a_report(tmp_path, stub_server, monkeypatch):
    url, _state = stub_server
    path = make_manifest(tmp_path)
    output = tmp_path / "report"
    monkeypatch.setenv("VOICEBOX_CLIENT_KEY", "private-client")
    monkeypatch.delenv("VOICEBOX_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "speech_benchmark.py",
            str(path),
            "--url",
            url,
            "--output",
            str(output),
            "--concurrency",
            "1",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
    )
    assert benchmark.main() == 0
    report = json.loads((output / "report.json").read_text())
    assert report["ok"]
    assert (output / "requests.csv").exists()
    assert (output / "summaries.csv").exists()
    assert (output / "cancellations.csv").exists()
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.main()
    assert (output / "report.json").read_bytes() == before


def test_corpus_wer_uses_total_reference_words_instead_of_mean_clip_wer():
    base = {
        "kind": "stt",
        "model": "stub",
        "source": "real",
        "voice": None,
        "concurrency": 1,
        "ok": True,
        "status": 200,
        "audio_s": 1,
        "latency_s": 1,
        "rtf": 1,
    }
    rows = [
        {
            **base,
            **benchmark.accuracy(
                "one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten"
            ),
        },
        {**base, **benchmark.accuracy("one", "two")},
    ]
    assert benchmark.summarize(rows, 2, 1)["wer"] == pytest.approx(1 / 11)
